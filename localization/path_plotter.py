import rclpy
from rclpy.node import Node
from nav_msgs.msg import Path
from geometry_msgs.msg import PoseStamped
from tf_transformations import quaternion_from_euler
import time

class PathPublisher(Node):
    def __init__(self):
        super().__init__('path_publisher')
        self.publisher_ = self.create_publisher(Path, '/robot_path', 10)
        self.path_msg = Path()
        self.path_msg.header.frame_id = 'map'

        # Create TF listener and buffer
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # Timer to periodically fetch the transform and publish the path
        self.timer = self.create_timer(0.1, self.timer_callback)

    def timer_callback(self):
        try:
            # Get the latest transform between 'map' and 'base_link' (or whatever frame you're using)
            transform = self.tf_buffer.lookup_transform('map', 'base_link', rclpy.time.Time())

            # Create a PoseStamped message from the transform
            pose = PoseStamped()
            pose.header.stamp = self.get_clock().now().to_msg()
            pose.header.frame_id = 'map'

            # Extract position and orientation from the transform
            pose.pose.position.x = transform.transform.translation.x
            pose.pose.position.y = transform.transform.translation.y
            pose.pose.position.z = transform.transform.translation.z

            pose.pose.orientation = transform.transform.rotation

            # Add the pose to the path message
            self.path_msg.poses.append(pose)

            # # Optionally, limit the number of poses in the path
            # if len(self.path_msg.poses) > 100:  # limit size of path
            #     self.path_msg.poses.pop(0)

            # Publish the path message
            self.publisher_.publish(self.path_msg)

        except Exception as e:
            self.get_logger().warn(f"Could not get transform: {e}")

def main(args=None):
    rclpy.init(args=args)
    node = PathPublisher()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
