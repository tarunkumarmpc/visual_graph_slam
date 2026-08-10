#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
import math
from sensor_msgs.msg import NavSatFix, Imu
from nav_msgs.msg import Path
from geometry_msgs.msg import PoseStamped, Quaternion, TransformStamped
from tf2_ros import TransformBroadcaster

def geodetic_to_enu(lat, lon, alt, lat0, lon0, alt0):
    a = 6378137.0
    f = 1.0 / 298.257223563
    b = a * (1 - f)
    e2 = 1.0 - (b / a) ** 2

    def N_rad(lat_rad):
        return a / math.sqrt(1.0 - e2 * math.sin(lat_rad) ** 2)

    lat_r = math.radians(lat)
    lon_r = math.radians(lon)
    lat0_r = math.radians(lat0)
    lon0_r = math.radians(lon0)

    N = N_rad(lat_r)
    X = (N + alt) * math.cos(lat_r) * math.cos(lon_r)
    Y = (N + alt) * math.cos(lat_r) * math.sin(lon_r)
    Z = (N * (1 - e2) + alt) * math.sin(lat_r)

    N0 = N_rad(lat0_r)
    X0 = (N0 + alt0) * math.cos(lat0_r) * math.cos(lon0_r)
    Y0 = (N0 + alt0) * math.cos(lat0_r) * math.sin(lon0_r)
    Z0 = (N0 * (1 - e2) + alt0) * math.sin(lat0_r)

    dX = X - X0
    dY = Y - Y0
    dZ = Z - Z0

    sin_lat0 = math.sin(lat0_r)
    cos_lat0 = math.cos(lat0_r)
    sin_lon0 = math.sin(lon0_r)
    cos_lon0 = math.cos(lon0_r)

    e = -sin_lon0 * dX + cos_lon0 * dY
    n = -sin_lat0 * cos_lon0 * dX - sin_lat0 * sin_lon0 * dY + cos_lat0 * dZ
    u = cos_lat0 * cos_lon0 * dX + cos_lat0 * sin_lon0 * dY + sin_lat0 * dZ

    return e, n, u


def quat_inv(q):
    """ Inverse of a quaternion [x, y, z, w] """
    return [-q[0], -q[1], -q[2], q[3]]

def quat_mult(q1, q2):
    """ Multiply two quaternions [x, y, z, w] """
    x1, y1, z1, w1 = q1
    x2, y2, z2, w2 = q2
    return [
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2,
        w1*w2 - x1*x2 - y1*y2 - z1*z2
    ]

def quat_rot(q, v):
    """ Rotate vector v by quaternion q """
    q_v = [v[0], v[1], v[2], 0.0]
    q_inv = quat_inv(q)
    res = quat_mult(quat_mult(q, q_v), q_inv)
    return [res[0], res[1], res[2]]


class GpsVisualizerNode(Node):
    def __init__(self):
        super().__init__('gps_visualizer_node')

        self.path_pub = self.create_publisher(Path, '/slam/gt_path', 10)
        self.gps_sub = self.create_subscription(NavSatFix, '/kitti/oxts/gps/fix', self.gps_callback, 100)
        self.imu_sub = self.create_subscription(Imu, '/kitti/oxts/imu', self.imu_callback, 100)

        self.latest_imu_q = Quaternion(x=0.0, y=0.0, z=0.0, w=1.0)
        # Guard: track whether at least one real IMU message has arrived.
        # If GPS fires first, q0_inv would be captured from the identity default,
        # leaving the GT trajectory in raw ENU frame instead of vehicle-aligned frame.
        self.has_received_imu = False
        self.origin = None
        self.path_msg = Path()
        self.path_msg.header.frame_id = 'map'

        self.tf_broadcaster = TransformBroadcaster(self)

        self.get_logger().info('GPS Visualizer Node Started. Waiting for first IMU + GPS...')

    def imu_callback(self, msg: Imu):
        # Cache the latest IMU orientation to fuse with the slower GPS.
        self.latest_imu_q = msg.orientation
        self.has_received_imu = True

    def gps_callback(self, msg: NavSatFix):
        if msg.status.status < 0:
            return  # No fix

        # CRITICAL: Do not set the ENU origin until we have a real IMU orientation.
        # Without this guard, q0_inv = identity (the default), and the GT trajectory
        # is published in raw ENU frame (East=+X, North=+Y) instead of the vehicle-
        # aligned frame (forward=+X, left=+Y), causing a ~27-degree heading mismatch
        # that makes all turns appear to go the wrong direction.
        if not self.has_received_imu:
            self.get_logger().warn('GPS arrived before first IMU — skipping until IMU is ready.', throttle_duration_sec=1.0)
            return

        if self.origin is None:
            self.origin = (msg.latitude, msg.longitude, msg.altitude)
            # Capture the very first IMU orientation to align ENU with the vehicle frame.
            # q0 is the vehicle heading in ENU at t=0. q0_inv rotates ENU coords into
            # the vehicle-forward frame so that SLAM and GT share the same reference frame.
            q0 = [
                self.latest_imu_q.x,
                self.latest_imu_q.y,
                self.latest_imu_q.z,
                self.latest_imu_q.w
            ]
            self.q0_inv = quat_inv(q0)
            self.get_logger().info(
                f'Set ENU origin: lat={msg.latitude:.6f}, lon={msg.longitude:.6f}, alt={msg.altitude:.2f} '
                f'| Initial IMU yaw captured (q0=[{q0[0]:.3f},{q0[1]:.3f},{q0[2]:.3f},{q0[3]:.3f}])')

        e, n, u = geodetic_to_enu(
            msg.latitude, msg.longitude, msg.altitude,
            self.origin[0], self.origin[1], self.origin[2]
        )

        # Rotate the ENU translation into the initial SLAM map frame
        p_map = quat_rot(self.q0_inv, [e, n, u])

        # Rotate the current IMU orientation into the initial SLAM map frame
        q_curr = [
            self.latest_imu_q.x,
            self.latest_imu_q.y,
            self.latest_imu_q.z,
            self.latest_imu_q.w
        ]
        q_map = quat_mult(self.q0_inv, q_curr)

        pose = PoseStamped()
        pose.header = msg.header
        pose.header.frame_id = 'map'
        
        pose.pose.position.x = float(p_map[0])
        pose.pose.position.y = float(p_map[1])
        pose.pose.position.z = float(p_map[2])

        pose.pose.orientation.x = float(q_map[0])
        pose.pose.orientation.y = float(q_map[1])
        pose.pose.orientation.z = float(q_map[2])
        pose.pose.orientation.w = float(q_map[3])

        self.path_msg.poses.append(pose)
        
        # Keep path header up to date
        self.path_msg.header.stamp = msg.header.stamp

        # Publish the path
        self.path_pub.publish(self.path_msg)

        # Broadcast the Transform map -> gt_base_link
        t = TransformStamped()
        t.header.stamp = msg.header.stamp
        t.header.frame_id = 'map'
        t.child_frame_id = 'gt_base_link'
        
        t.transform.translation.x = pose.pose.position.x
        t.transform.translation.y = pose.pose.position.y
        t.transform.translation.z = pose.pose.position.z
        
        t.transform.rotation = pose.pose.orientation
        
        self.tf_broadcaster.sendTransform(t)


def main(args=None):
    rclpy.init(args=args)
    node = GpsVisualizerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()

if __name__ == '__main__':
    main()
