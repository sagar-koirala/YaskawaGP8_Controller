from coppeliasim_zmqremoteapi_client import RemoteAPIClient
import numpy as np
import cv2

class CoppeliaSimInterface:
    def __init__(self):
        self.client = None
        self.sim = None
        self.robot_handle = None
        self.joint_handles = []
        self.end_effector_handle = None
        self.gripper_handle = None
        self.gripper_motor1 = None
        self.gripper_motor2 = None
        self.closing_vel = 0.15
        self.camera_handle = None

    def connect(self):
        try:
            self.client = RemoteAPIClient()
            self.sim = self.client.getObject('sim')
            if self.sim.getSimulationState() == self.sim.simulation_stopped:
                self.sim.startSimulation()
            self.robot_handle = self.sim.getObject('/yaskawa')
            self.end_effector_handle = self.sim.getObject('/yaskawa/gripperEF')
            self.gripper_handle = self.sim.getObject('/yaskawa/MicoHand')
            self.gripper_motor1 = self.sim.getObject('/yaskawa/MicoHand/fingers12_motor1')
            self.gripper_motor2 = self.sim.getObject('/yaskawa/MicoHand/fingers12_motor2')
            for i in range(6):
                handle = self.sim.getObject(f'/yaskawa/joint{i+1}')
                self.joint_handles.append(handle)
            return True
        except Exception as e:
            print(f"Connection failed: {e}")
            return False

    def disconnect(self):
        try:
            if self.sim is not None:
                self.sim.stopSimulation()
            # Reset all connection-related attributes
            self.sim = None
            self.client = None
            self.joint_handles = []
            self.robot_handle = None
            self.end_effector_handle = None
            self.gripper_handle = None
            self.gripper_motor1 = None
            self.gripper_motor2 = None
            self.camera_handle = None
            return True
        except Exception as e:
            print(f"Error disconnecting: {e}")
            return False

    def is_connected(self):
        return self.sim is not None

    def get_joint_angles(self):
        if not self.is_connected():
            return None
        angles = []
        for handle in self.joint_handles:
            angle_rad = self.sim.getJointPosition(handle)
            angles.append(np.degrees(angle_rad))
        return angles

    def set_joint_angles(self, angles_deg):
        if not self.is_connected():
            return False
        try:
            for handle, angle in zip(self.joint_handles, angles_deg):
                self.sim.setJointTargetPosition(handle, np.radians(angle))
            return True
        except Exception as e:
            print(f"Failed to set joint angles: {e}")
            return False

    def get_end_effector_pose(self):
        if not self.is_connected():
            return None, None
        try:
            pos = self.sim.getObjectPosition(self.end_effector_handle, -1)
            orient = self.sim.getObjectOrientation(self.end_effector_handle, -1)
            orient_deg = [np.degrees(angle) for angle in orient]
            return pos, orient_deg
        except Exception as e:
            print(f"Failed to get end effector pose: {e}")
            return None, None

    def get_gripper_status(self):
        if not self.is_connected():
            return None
        try:
            velocity = self.sim.getJointTargetVelocity(self.gripper_motor1)
            status = 1 if velocity > 0 else 0
            return status
        except Exception as e:
            print(f"Failed to get gripper status: {e}")
            return None

    def set_gripper_status(self, status):
        if not self.is_connected():
            return False
        try:
            velocity = self.closing_vel if status == 1 else -self.closing_vel
            self.sim.setJointTargetVelocity(self.gripper_motor1, velocity)
            self.sim.setJointTargetVelocity(self.gripper_motor2, velocity)
            return True
        except Exception as e:
            print(f"Failed to set gripper status: {e}")
            return False

    def get_camera_image(self, camera_handle=None):
        if not self.is_connected():
            return None, None
        
        if camera_handle is None:
            camera_handle = self.camera_handle
        
        try:
            img, [resX, resY] = self.sim.getVisionSensorImg(camera_handle)
            img = np.frombuffer(img, dtype=np.uint8).reshape(resY, resX, 3)
            img = cv2.flip(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), 0)
            return img, (resX, resY)
        except Exception as e:
            print(f"Failed to get camera image: {e}")
            return None, None

    def get_camera_parameters(self, camera_handle=None):
        if not self.is_connected():
            return None, None
        
        if camera_handle is None:
            camera_handle = self.camera_handle
        
        try:
            resolution = self.sim.getVisionSensorResolution(camera_handle)
            fov = self.sim.getObjectFloatParam(camera_handle, 1004)  # 1004 is perspective angle parameter
            return resolution, fov
        except Exception as e:
            print(f"Failed to get camera parameters: {e}")
            return None, None
            
    def create_or_update_dummy(self, name, position=None, orientation=None, reference_handle=-1, 
                              quaternion=None, size=0.05, visible=True, existing_handle=None):
        """Create a new dummy or update an existing one with the specified properties"""
        if not self.is_connected():
            return None
            
        try:
            # Create or use existing handle
            handle = existing_handle
            if handle is None:
                handle = self.sim.createDummy(size)
                self.sim.setObjectName(handle, name)
            # Set position if provided
            if position is not None:
                self.sim.setObjectPosition(handle, reference_handle, position)
            # Set orientation if provided
            if orientation is not None:
                self.sim.setObjectOrientation(handle, reference_handle, orientation)
            # Set quaternion if provided
            if quaternion is not None:
                self.sim.setObjectQuaternion(handle, reference_handle, quaternion)
            # Set visibility
            if visible:
                self.sim.setObjectInt32Param(handle, self.sim.objintparam_visibility_layer, 1)
            else:
                self.sim.setObjectInt32Param(handle, self.sim.objintparam_visibility_layer, 0)
            return handle
        except Exception as e:
            print(f"Failed to create or update dummy: {e}")
            return None
            
    # Keep these utility functions as they encapsulate common operations
    def get_object_position(self, handle, reference_handle=-1):
        """Get the position of an object relative to a reference frame"""
        if not self.is_connected():
            return None
        try:
            return self.sim.getObjectPosition(handle, reference_handle)
        except Exception as e:
            print(f"Failed to get object position: {e}")
            return None
    
    def set_joint_velocities(self, velocities):
        """
        Set the velocity of each joint.
        
        Args:
            velocities (list): List of joint velocities in degrees/second
        
        Returns:
            bool: Success or failure
        """
        if not self.is_connected():
            return False
        try:
            for handle, velocity in zip(self.joint_handles, velocities):
                # Convert degrees/second to radians/second
                velocity_rad = np.radians(velocity)
                self.sim.setJointTargetVelocity(handle, velocity_rad)
            return True
        except Exception as e:
            print(f"Failed to set joint velocities: {e}")
            return False
    
    def get_joint_velocities(self):
        """
        Get the current velocity of each joint.
        
        Returns:
            list: List of joint velocities in degrees/second or None if failed
        """
        if not self.is_connected():
            return None
        velocities = []
        try:
            for handle in self.joint_handles:
                velocity_rad = self.sim.getJointVelocity(handle)
                velocities.append(np.degrees(velocity_rad))
            return velocities
        except Exception as e:
            print(f"Failed to get joint velocities: {e}")
            return None
        
    def get_object_orientation(self, handle, reference_handle=-1):
        """Get the orientation of an object relative to a reference frame"""
        if not self.is_connected():
            return None
        try:
            orient = self.sim.getObjectOrientation(handle, reference_handle)
            orient_deg = [np.degrees(angle) for angle in orient]
            return orient_deg
        except Exception as e:
            print(f"Failed to get object orientation: {e}")
            return None
            
    def get_object_quaternion(self, handle, reference_handle=-1):
        """Get the quaternion of an object relative to a reference frame"""
        if not self.is_connected():
            return None
        try:
            return self.sim.getObjectQuaternion(handle, reference_handle)
        except Exception as e:
            print(f"Failed to get object quaternion: {e}")
            return None
            
    def set_object_quaternion(self, handle, reference_handle, quaternion):
        """Set the quaternion of an object relative to a reference frame"""
        if not self.is_connected():
            return False
        try:
            self.sim.setObjectQuaternion(handle, reference_handle, quaternion)
            return True
        except Exception as e:
            print(f"Failed to set object quaternion: {e}")                  
            return False
            
    def set_object_visibility(self, handle, visible, layer=1):
        """Set the visibility of an object"""
        if not self.is_connected():
            return False
        try:
            self.sim.setObjectInt32Param(handle, self.sim.objintparam_visibility_layer, layer if visible else 0)
            return True
        except Exception as e:
            print(f"Failed to set object visibility: {e}")
            return False