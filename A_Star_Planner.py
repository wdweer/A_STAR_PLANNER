from a_star import AStarPlanner
import rospy
import matplotlib.pyplot as plt
from geometry_msgs.msg import PoseStamped
from autoware_msgs.msg import DetectedObjectArray
import numpy as np
from hmcl_msgs.msg import LaneArray

class A_Star_Planner():
    def __init__(self):
        rospy.init_node("A_Star_Planner")
        self.current_pose_sub = rospy.Subscriber('/current_pose', PoseStamped, self.current_pose_callback)
        self.detected_object_sub = rospy.Subscriber('/tracking/car',DetectedObjectArray, self.object_callback)
        self.global_traj_sub=rospy.Subscriber('/global_traj',LaneArray,self.global_traj_callback)
        rate = rospy.Rate(10)
        ego_x=0.0
        ego_y=0.0
        self.ego_yaw = 0.0  # Placeholder initial value
        self.cx = []
        self.cy = []
        self.global_points=[]
        while not rospy.is_shutdown():
            rate.sleep()
            
    def global_traj_callback(self,data):
        for j in data.lanes[1].waypoints:
            self.cx.append(j.pose.pose.position.x)
            self.cy.append(j.pose.pose.position.y)
        for i in range(len(self.cx)):
            self.global_points.append([self.cx[i],self.cy[i]])
        
        self.lane_data=data.lanes
        
    def current_pose_callback(self,data):
        self.ego_x=data.pose.position.x
        self.ego_y=data.pose.position.y
    def cartesian_to_frenet(self, centerline, point):
    # 중심선의 아크 길이 계산
        centerline = np.array(centerline)
        diffs = np.diff(centerline, axis=0)
        dists = np.linalg.norm(diffs, axis=1)
        arclength = np.insert(np.cumsum(dists), 0, 0.0)

        # 점에서 각 선분까지의 거리 계산 및 최소 거리 찾기
        point = np.array(point)
        min_dist = float('inf')
        s, l = 0, 0

        for i in range(len(diffs)):
            p1 = centerline[i]
            p2 = centerline[i + 1]

            # 점과 선분 사이의 수직 거리 계산
            line_vec = p2 - p1
            point_vec = point - p1
            line_len = np.linalg.norm(line_vec)
            proj_length = np.dot(point_vec, line_vec) / line_len
            proj_point = p1 + (proj_length / line_len) * line_vec
            
            dist = np.linalg.norm(point - proj_point)
            if dist < min_dist:
                min_dist = dist
                s = arclength[i] + proj_length
                l = dist
        
        return s, l
      
    def frenet_to_cartesian(self, centerline, s, l):
    # 중심선의 아크 길이 계산
        centerline = np.array(centerline)
        diffs = np.diff(centerline, axis=0)
        dists = np.linalg.norm(diffs, axis=1)
        arclength = np.insert(np.cumsum(dists), 0, 0.0)

        # s에 해당하는 세그먼트 인덱스 찾기
        segment_index = np.searchsorted(arclength, s) - 1
        if segment_index < 0:
            segment_index = 0
        elif segment_index >= len(centerline) - 1:
            segment_index = len(centerline) - 2

        p1 = centerline[segment_index]
        p2 = centerline[segment_index + 1]

        # 세그먼트의 방향 벡터 및 단위 벡터 계산
        segment_vector = p2 - p1
        segment_length = dists[segment_index]
        segment_unit_vector = segment_vector / segment_length

        # s에 대한 기본점 계산
        base_point = p1 + segment_unit_vector * (s - arclength[segment_index])

        # 법선 벡터 계산 (세그먼트에 수직)
        normal_vector = np.array([-segment_unit_vector[1], segment_unit_vector[0]])

        # 카르테시안 좌표 계산
        cartesian_point = base_point + normal_vector * l

        return cartesian_point[0], cartesian_point[1]
        
    def object_callback(self,data):
        rospy.loginfo("DetectedObjectArray received")
        self.target_veh_dic_x={}
        self.target_veh_dic_y={}
        self.obj=[]
        self.objects_data=data
        for obj in data.objects:
            self.target_velocity_x=obj.velocity.linear.x
            self.target_velocity_y=obj.velocity.linear.y
            self.target_x = obj.pose.position.x
            self.target_y = obj.pose.position.y
            self.target_orientation_x=obj.pose.orientation.x 
            self.target_orientation_y=obj.pose.orientation.y 
            self.target_orientation_z=obj.pose.orientation.z 
            self.target_orientation_w=obj.pose.orientation.w
            self.target_yaw = obj.velocity.angular.z
            self.target_velocity=(self.target_velocity_x**2+self.target_velocity_y**2)**(1/2)
            self.target_yaw=math.atan2(self.target_velocity_y,self.target_velocity_x )
            self.obj.append(obj.label)
            self.detected_object_heuristic(obj.label)
        self.main()
            
    def detected_object_heuristic(self,obj):
        model_prediction_x = self.target_x
        target_velocity_x = self.target_velocity_x
        target_yaw = self.target_yaw
        model_prediction_y = self.target_y
        target_velocity_y = self.target_velocity_y
        target_angular_z = self.target_angular_z
        self.model_prediction_x = []
        self.model_prediction_y = []
        for i in range(self.model_predicted_num):
            self.model_prediction_x.append(model_prediction_x)
            model_prediction_x += target_velocity_x
            target_yaw += target_angular_z
            target_velocity_x = self.target_velocity * math.cos(target_yaw)
        self.target_veh_dic_x[obj]=self.model_prediction_x
        for j in range(self.model_predicted_num):
            self.model_prediction_y.append(model_prediction_y)
            model_prediction_y += target_velocity_y
            target_yaw += target_angular_z
            target_velocity_y = self.target_velocity * math.sin(target_yaw)
        self.target_veh_dic_y[obj]=self.model_prediction_y
        
    def main(self):
        print(__file__ + " start!!")
        show_animation = True

        # start and goal position
        sx = self.ego_x  # [m]
        sy = self.ego_y
        print(len(self.cx))
        ss,sd = self.cartesian_to_frenet(self.global_points,[sx,sy])
        gs=50+ss
        gd=sd
        gx,gy= self.frenet_to_cartesian(self.global_points,gs,gd)
        avante_x=10
        avante_y=10
        # [m]
        # gx = 50.0  # [m]
        # gy = 50.0  # [m]
        grid_size = 2.0  # [m]
        robot_radius = 1.0  # [m]

        # set obstacle positions
        ox, oy = [], []
        for i in range(-2500,2500):
            ox.append(i)
            oy.append(2500)
        for i in range(-2500,2500):
            ox.append(i)
            oy.append(-2500)
            
        for i in range(-2500,2500):
            ox.append(2500)
            oy.append(i)
        for i in range(-2500,2500):
            ox.append(-2500)
            oy.append(i)
        # for i in range(-10, 60):
        #     ox.append(i)
        #     oy.append(-10.0)
        # for i in range(-10, 60):
        #     ox.append(60.0)
        #     oy.append(i)
        # for i in range(-10, 61):
        #     ox.append(i)
        #     oy.append(60.0)
        # for i in range(-10, 61):
        #     ox.append(-10.0)
        #     oy.append(i)
        # for i in range(-10, 40):
        #     ox.append(20.0)
        #     oy.append(i)
        # for i in range(0, 40):
        #     ox.append(40.0)
        #     oy.append(60.0 - i)
        for obj in self.obj:
            for i in range(-avante_x/2,avante_x/2):
                ox.append(i+self.target_veh_dic_x[obj][0])
                oy.append(avante_y/2+self.target_veh_dic_y[obj][0])
            for i in range(-avante_x/2,avante_x/2):
                ox.append(avante_x/2+self.target_veh_dic_x[obj][0])
                oy.append(i+self.target_veh_dic_y[obj][0])
            for i in range(-avante_x/2,avante_x/2):
                ox.append(i+self.target_veh_dic_x[obj][0])
                oy.append(-avante_y/2+self.target_veh_dic_y[obj][0])
            for i in range(-avante_x/2,avante_x/2):
                ox.append(-avante_x/2+self.target_veh_dic_x[obj][0])
                oy.append(i+self.target_veh_dic_y[obj][0])

        a_star = AStarPlanner(ox, oy, grid_size, robot_radius)
        rx, ry = a_star.planning(sx, sy, gx, gy)
if __name__ == '__main__':
    try:
        A_Star_Planner()
    except rospy.ROSInterruptException:
        pass