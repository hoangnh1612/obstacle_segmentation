#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/msg/laser_scan.hpp"
#include <iostream>
#include <Eigen/Dense>
#include <vector>
#include <cmath>
#include <utility>
#include"obstacles.h"
#include "nav_msgs/msg/odometry.hpp"
// include yf2 for tf2::getYaw::
#include "tf2_geometry_msgs/tf2_geometry_msgs.h"
#include "tf2/utils.h"



using namespace teb_local_planner;
inline cv::Point scalePoint(double x, double y, double scale, double offset_x, double offset_y)
{
    int px = static_cast<int>(x * scale + offset_x);
    int py = static_cast<int>(y * scale + offset_y);
    return cv::Point(px, py);
}

double findDifferenceOrientation(double angle1, double angle2)
{
    double diff = angle1 - angle2;
    while (diff > M_PI)
        diff -= 2 * M_PI;
    while (diff < -M_PI)
        diff += 2 * M_PI;
    return diff;
}

struct RobotPose
{
    double x;
    double y;
    double theta;
};

class TestClustering
{
public:
    TestClustering(const LaserScanData& laser_scan_)
        : laser_scan(laser_scan_), angular_resolution(M_PI / 180.0) // Example angular resolution: 1 degree
    {
        obstacles = new ObstContainer();
        all_points = new ObstContainer();
    }

    ~TestClustering()
    {
        delete obstacles;
        delete all_points;
    }



    void clusterPointCloud()
    {
        point_cloud_clusters.clear();

        double lambda = M_PI / 4; // Acceptable angle for determining cluster membership
        const double omega_r = 0.1; // Standard deviation of the noise of the distance measure

        LaserPointCloud point_block;
        LaserScanData point_list;

        if (laser_scan.empty())
            return;

        // Initialize with the first point
        point_list.push_back(laser_scan.front());
        point_block.push_back(laser_scan.front().point);

        for (size_t i = 1; i < laser_scan.size(); ++i)
        {
            // Distance between current point and the last point in the cluster
            double distance = (laser_scan[i].point - point_list.back().point).norm();
            // Delta theta between current point and the last point in the cluster
            double dTheta = laser_scan[i].angle - point_list.back().angle;

            // Calculate the distance threshold
            double angle_diff = std::abs(findDifferenceOrientation(laser_scan[i].angle, point_list.back().angle));
            double D_thd;
            if (angle_diff > 4 * std::abs(angular_resolution))
                D_thd = 0.0;
            else
                D_thd = std::min(point_list.back().range, laser_scan[i].range) * std::sin(dTheta) / std::sin(lambda - dTheta) + 3 * omega_r;

            if (distance < D_thd)
            {
                // Same cluster
                point_list.push_back(laser_scan[i]);
                point_block.push_back(laser_scan[i].point);
            }
            else
            {
                // New cluster
                if (!point_block.empty())
                {
                    point_cloud_clusters.push_back(point_block);
                }
                point_block.clear();
                point_list.clear();
                point_list.push_back(laser_scan[i]);
                point_block.push_back(laser_scan[i].point);
            }
        }

        // Add the last cluster
        if (!point_block.empty())
        {
            point_cloud_clusters.push_back(point_block);
        }
    }

    void printClusters() const
    {
        std::cout << "Number of clusters: " << point_cloud_clusters.size() << std::endl;
        for (size_t i = 0; i < point_cloud_clusters.size(); ++i)
        {
            std::cout<<point_cloud_clusters[i].front()<<std::endl;
            std::cout << "Cluster " << i + 1 << " (" << point_cloud_clusters[i].size() << " points):" << std::endl;
            for (const auto& point : point_cloud_clusters[i])
            {
                std::cout << "(" << point.x() << ", " << point.y() << ") ";
            }
            std::cout << std::endl;
        }
    }
    bool isConvexObject(LaserPointCloud cluster)
    {
        double left_dis = cluster.front().norm() - extra_distance;
        double right_dis = cluster.back().norm() - extra_distance;
        double mid_dis = 0;
        for (int i = 1; i< cluster.size() - 1; i++)
        {
            mid_dis += cluster[i].norm();
        }
        mid_dis = mid_dis / (cluster.size() - 2);
        if (mid_dis < left_dis && mid_dis < right_dis)
            return true;
        else
            return false;
    }

    void rectangleFitting()
    {
        // Clear the obstacles
        rectangles.clear();
        obstacles->clear();
        all_points->clear();
        for (size_t i = 0; i < point_cloud_clusters.size(); i++)
        {
            for (size_t j = 0; j < point_cloud_clusters[i].size(); j++)
            {
                all_points->push_back(ObstaclePtr(new PointObstacle(point_cloud_clusters[i][j])));
            }
        }

        for (size_t i = 0; i < point_cloud_clusters.size(); i++)
        {
            if (isConvexObject(point_cloud_clusters[i]))
            {
                size_t n = point_cloud_clusters[i].size();
                Eigen::VectorXd e1(2), e2(2);
                Eigen::MatrixXd X(n, 2);

                for (size_t j = 0; j < n; j++)
                {
                    X(j, 0) = point_cloud_clusters[i][j].x();
                    X(j, 1) = point_cloud_clusters[i][j].y();
                }

                Eigen::VectorXd C1(n), C2(n);
                double q;
                double theta = 0.0;
                double step = M_PI / (2.0 * step_of_theta);
                Eigen::MatrixXd Q(step_of_theta, 2); // Sử dụng MatrixXd thay vì ArrayX2d

                for (int k = 0; k < step_of_theta; ++k)
                {
                    e1 << cos(theta), sin(theta);
                    e2 << -sin(theta), cos(theta);
                    C1 = X * e1;
                    C2 = X * e2;
                    q = closenessCriterion(C1, C2, 0.0001) + areaCriterion(C1, C2) + varianceCriterion(C1,C2);
                    Q(k, 0) = theta;
                    Q(k, 1) = q;

                    theta += step;
                }

                Eigen::MatrixXd::Index max_index;
                Q.col(1).maxCoeff(&max_index); // find Q with maximum value
                theta = Q(max_index, 0);
                e1 << cos(theta), sin(theta);
                e2 << -sin(theta), cos(theta);
                C1 = X * e1;
                C2 = X * e2;

                double a1 = cos(theta);
                double b1 = sin(theta);
                double c1 = -C1.minCoeff(); // Đảo dấu

                double a2 = -sin(theta);
                double b2 = cos(theta);
                double c2 = -C2.minCoeff(); // Đảo dấu

                double a3 = cos(theta);
                double b3 = sin(theta);
                double c3 = C1.maxCoeff(); // Giữ nguyên

                double a4 = -sin(theta);
                double b4 = cos(theta);
                double c4 = C2.maxCoeff(); // Giữ nguyên

                std::vector<Eigen::Vector2d> corners;
                Eigen::Vector2d p1 = lineIntersection(a1, b1, c1, a2, b2, c2);
                Eigen::Vector2d p2 = lineIntersection(a2, b2, c2, a3, b3, c3);
                Eigen::Vector2d p3 = lineIntersection(a3, b3, c3, a4, b4, c4);
                Eigen::Vector2d p4 = lineIntersection(a1, b1, c1, a4, b4, c4);

                // In ra các điểm giao nhau để kiểm tra
                std::cout << "Intersection points:" << std::endl;
                std::cout << "p1: (" << p1.x() << ", " << p1.y() << ")" << std::endl;
                std::cout << "p2: (" << p2.x() << ", " << p2.y() << ")" << std::endl;
                std::cout << "p3: (" << p3.x() << ", " << p3.y() << ")" << std::endl;
                std::cout << "p4: (" << p4.x() << ", " << p4.y() << ")" << std::endl;

                // Kiểm tra các điểm giao nhau hợp lệ
                if (!std::isnan(p1.x()) && !std::isnan(p1.y()) &&
                    !std::isnan(p2.x()) && !std::isnan(p2.y()) &&
                    !std::isnan(p3.x()) && !std::isnan(p3.y()) &&
                    !std::isnan(p4.x()) && !std::isnan(p4.y()))
                {
                    corners.push_back(p1);
                    corners.push_back(p2);
                    corners.push_back(p3);
                    corners.push_back(p4);
                    rectangles.push_back(corners);
                }
                else
                {
                    std::cerr << "Warning: Invalid intersection points detected. Skipping rectangle." << std::endl;
                }

                // Kiểm tra các cạnh của hình chữ nhật
                if (corners.size() == 4)
                {
                    double edge_1 = (corners[0] - corners[1]).norm();
                    double edge_2 = (corners[1] - corners[2]).norm();
                    if (std::min(edge_1, edge_2) == 0) // Tránh chia cho 0
                    {
                        std::cerr << "Warning: Zero length edge detected. Skipping." << std::endl;
                        continue;
                    }
                    if (std::max(edge_1, edge_2) / std::min(edge_1, edge_2) > 5)
                    {
                        std::vector<LineSegment> lineSegments = lineExtraction(point_cloud_clusters[i]);
                        std::cout << "Extracted " << lineSegments.size() << " line segments from cluster " << i + 1 << std::endl;
                        for (size_t k = 0; k < lineSegments.size(); k++)
                        {
                            rectangles.push_back(lineSegments[k]);
                            obstacles->push_back(ObstaclePtr(new LineObstacle(lineSegments[k][0], lineSegments[k][1])));
                        }
                    }
                    else
                    {
                        // Đã thêm vào rectangles ở trên
                        Point2dContainer vertices;
                        for (size_t k = 0; k < corners.size(); k++)
                        {
                            vertices.push_back(corners[k]);
                        }
                        obstacles->push_back(ObstaclePtr(new PolygonObstacle(vertices)));
                    }
                }
            }
            else
            {
                std::vector<LineSegment> lineSegments = lineExtraction(point_cloud_clusters[i]);
                std::cout << "Extracted " << lineSegments.size() << " line segments from cluster " << i + 1 << std::endl;
                for (size_t k = 0; k < lineSegments.size(); k++)
                {
                    rectangles.push_back(lineSegments[k]);
                    obstacles->push_back(ObstaclePtr(new LineObstacle(lineSegments[k][0], lineSegments[k][1])));
                }
            }
        }

        for (size_t i = 0; i < rectangles.size(); ++i)
        {
            std::cout << "Rectangle " << i + 1 << " has " << rectangles[i].size() << " points:" << std::endl;
            for (size_t j = 0; j < rectangles[i].size(); ++j)
            {
                std::cout << "(" << rectangles[i][j].x() << ", " << rectangles[i][j].y() << ") ";
            }
            std::cout << std::endl;
        }

    }
    std::vector<LineSegment> lineExtraction(LaserPointCloud cluster)
    {

        std::vector<LineSegment> line_segments;
        if (cluster.size() < MINIMUM_POINTS_CHECK) return line_segments;

        // 1#: we initial a line from start to end
        //-----------------------------------------
        Eigen::Vector2d start = cluster.front();
        Eigen::Vector2d end = cluster.back();
        LineIndex l;
        l.first = 0;
        l.second = cluster.size() - 1;
        std::list<LineIndex> line_list;
        line_list.push_back(l);

        while (!line_list.empty()) 
        {
            // 2#: every time we take the first line in
            //line list to check if every point is on this line
            //-----------------------------------------
            LineIndex& lr = *line_list.begin();

            //
            if (lr.second - lr.first < MINIMUM_INDEX || lr.first == lr.second)
            {
                line_list.pop_front();
                continue;
            }

            // 3#: use two points to generate a line equation
            //-----------------------------------------
            start.x() = cluster[lr.first].x();
            start.y() = cluster[lr.first].y();
            end.x() = cluster[lr.second].x();
            end.y() = cluster[lr.second].y();

            // two points P1(x1, y1), P2(x2,y2) are given, and these two points are not the same
            // we can calculate an equation to model a line these two points are on.
            // A = y2 - y1
            // B = x1 - x2
            // C = x2 * y1 - x1 * y2
            double A = end.y() - start.y();
            double B = start.x() - end.x();
            double C = end.x() * start.y() - start.x() * end.y();

            double max_distance = 0;
            int max_i;
            int gap_i(-1);
            // the kernel code
            for (int i = lr.first + 1; i <= lr.second - 1; i++) 
            {
                // 4#: if two points' distance is too large, it's meaningless to generate a line
                // connects these two points, so we have to filter it.
                //-----------------------------------------
                double point_gap_dist = hypot(cluster[i].x() - cluster[i+1].x(), cluster[i].y() - cluster[i+1].y());
                if (point_gap_dist > MAXIMUM_GAP_DISTANCE) 
                {
                    gap_i = i;
                    break;
                }

                // 5#: calculate the distance between every point to the line
                //-----------------------------------------
                double dist = fabs(A * cluster[i].x() + B * cluster[i].y() + C) / hypot(A, B);
                if (dist > max_distance) 
                {
                    max_distance = dist;
                    max_i = i;
                }
            }

            // 6#: if gap is too large or there's a point is far from the line,
            // we have to split this line to two line, then check again.
            //-----------------------------------------
            if (gap_i != -1) 
            {
                int tmp = lr.second;
                lr.second = gap_i;
                LineIndex ll;
                ll.first = gap_i + 1;
                ll.second = tmp;
                line_list.insert(++line_list.begin(), ll);
            }
            else if (max_distance > IN_LINE_THRESHOLD) 
            {
                int tmp = lr.second;
                lr.second = max_i;
                LineIndex ll;
                ll.first = max_i + 1;
                ll.second = tmp;
                line_list.insert(++line_list.begin(), ll);
            } 
            else 
            {
                LineSegment line_;
                line_.push_back(cluster[line_list.front().first]);
                line_.push_back(cluster[line_list.front().second]);
                line_segments.push_back(line_);
                line_list.pop_front();
            }
        }
        return line_segments;
    }
    void visualize()
    {

    }
    Eigen::Vector2d lineIntersection(double a1, double b1, double c1, double a2, double b2, double c2)
    {
        double determinant = a1*b2 - a2*b1;
        Eigen::Vector2d intersection_point;
        intersection_point.x()  = (b2*c1 - b1*c2)/determinant;
        intersection_point.y() = (a1*c2 - a2*c1)/determinant;

        return intersection_point;
    }

    double areaCriterion(const Eigen::VectorXd &C1, const Eigen::VectorXd &C2)
    {
        double c1_max = C1.maxCoeff();
        double c1_min = C1.minCoeff();
        double c2_max = C2.maxCoeff();
        double c2_min = C2.minCoeff();

        double alpha = -(c1_max - c1_min) * (c2_max - c2_min);

        return alpha;
    }

    double closenessCriterion(const Eigen::VectorXd &C1, const Eigen::VectorXd &C2, const double &d0)
    {
        double c1_max = C1.maxCoeff();
        double c1_min = C1.minCoeff();
        double c2_max = C2.maxCoeff();
        double c2_min = C2.minCoeff();

        // Eigen::VectorXd C1_max = c1_max - C1.array(); 
        // Eigen::VectorXd C1_min = C1.array() - c1_min;
        // Eigen::VectorXd D1, D2;
        // if(C1_max.squaredNorm() < C1_min.squaredNorm()){
        //     D1 = C1_max;
        // }
        // else{
        //     D1 = C1_min;
        // }
        // Eigen::VectorXd C2_max = c2_max - C2.array(); 
        // Eigen::VectorXd C2_min = C2.array() - c2_min;
        // if(C2_max.squaredNorm() < C2_min.squaredNorm()){
        //     D2 = C2_max;
        // }
        // else{
        //     D2 = C2_min;
        // }

        // double d, min;
        // double b = 0 ;
        // for (int i = 0; i < D1.size(); ++i) 
        // {
        //     min = std::min(D1(i),D2(i));
        //     d = std::max(min, d0);
        //     b = b + 1/d;
        // }
        
        // return b; 


        Eigen::VectorXd d1 = (c1_max - C1.array()).cwiseMin(C1.array() - c1_min);
        Eigen::VectorXd d2 = (c2_max - C2.array()).cwiseMin(C2.array() - c2_min);

        double b = 0.0;
        for (int i = 0; i < d1.size(); ++i) 
        {
            double dd = std::min(d1(i), d2(i));
            dd = std::max(dd, d0);  
            b += 1.0 / dd;
        }
        return b;
 
    }
    double computeVariance(const Eigen::VectorXd &vec)
    {
        std::vector<double> values;
        values.reserve(vec.size());
        for (int i = 0; i < vec.size(); ++i)
        {
            if (vec(i) != 0.0)
            {
                values.push_back(vec(i));
            }
        }

        // Bước 2: nếu ít hơn 2 phần tử thì phương sai = 0
        if (values.size() < 2) 
            return 0.0;

        // Bước 3: tính trung bình
        double sum = 0.0;
        for (double v : values)
        {
            sum += v;
        }
        double mean = sum / values.size();

        // Bước 4: tính tổng bình phương độ lệch (x - mean)^2
        double sq_sum = 0.0;
        for (double v : values)
        {
            sq_sum += (v - mean) * (v - mean);
        }

        // Bước 5: chia cho (n - 1) để có sample variance
        double var = sq_sum / (values.size() - 1);

        return var;
    }

    double varianceCriterion(const Eigen::VectorXd &C1, const Eigen::VectorXd &C2)
    {
        double c1_max = C1.maxCoeff();
        double c1_min = C1.minCoeff();
        double c2_max = C2.maxCoeff();
        double c2_min = C2.minCoeff();
        Eigen::VectorXd d1 = (c1_max - C1.array()).cwiseMin(C1.array() - c1_min);
        Eigen::VectorXd d2 = (c2_max - C2.array()).cwiseMin(C2.array() - c2_min);

        // Tách e1, e2 (như Python code):
        Eigen::VectorXd e1(d1.size()), e2(d2.size());
        e1.setZero();
        e2.setZero();

        for (int i = 0; i < d1.size(); ++i) {
            if (d1(i) < d2(i))
                e1(i) = d1(i);
            else
                e2(i) = d2(i);
        }

        // Tính variance
        double var_e1 = computeVariance(e1); // cắt những phần tử = 0 nếu cần
        double var_e2 = computeVariance(e2);

        double gamma = -(var_e1 + var_e2);
        return gamma;

        // Eigen::VectorXd C1_max = c1_max - C1.array(); 
        // Eigen::VectorXd C1_min = C1.array() - c1_min;
        // Eigen::VectorXd D1, D2;
        // if(C1_max.squaredNorm() < C1_min.squaredNorm()){
        //     D1 = C1_max;
        // }
        // else{
        //     D1 = C1_min;
        // }
        // Eigen::VectorXd C2_max = c2_max - C2.array(); 
        // Eigen::VectorXd C2_min = C2.array() - c2_min;
        // if(C2_max.squaredNorm() < C2_min.squaredNorm()){
        //     D2 = C2_max;
        // }
        // else{
        //     D2 = C2_min;
        // }
        // Eigen::VectorXd E1(D1.size()), E2(D2.size());
        // for (int i = 0; i < E1.size(); i++)
        // {
        //     if (D1(i) < D2(i)) E1(i) = D1(i);
        //     if (D2(i) < D1(i)) E2(i) = D2(i);
        // }
        // double gamma = - sqrt((E1.array() - E1.mean()).square().sum() / (E1.size() - 1)) - sqrt((E2.array() - E2.mean()).square().sum() / (E2.size() - 1));
        // return gamma;
 
    }
    void robotVisualize(cv::Mat &map, RobotPose robot_pose)
    {
        // circle, 0.1m radius

        cv::Point center = scalePoint(robot_pose.x, robot_pose.y, scale, offset_x, offset_y);

        cv::circle(map, center, 15, cv::Scalar(0,0,255), -1);
        // draw robot orientation, represented by a line
        double angle = robot_pose.theta;
        cv::Point end;
        end.x = center.x + 20 * cos(angle);
        end.y = center.y + 20 * sin(angle);
        cv::line(map, center, end, cv::Scalar(0,0,255), 2);
    }

    void obsVisualize(cv::Mat &map, bool point_vis, RobotPose robot_pose)
    {
        robotVisualize(map, robot_pose);
        if (point_vis)
        {
            for(const auto& obst : *all_points)
            {
                // visualize point obstacle
                Eigen::Vector2d pos = boost::static_pointer_cast<PointObstacle>(obst)->position();
                cv::Point pc = scalePoint(pos.x(), pos.y(), scale, offset_x, offset_y);
                cv::circle(map, pc, 5, cv::Scalar(0,0,0), -1);
            }
        }
        else
        {
            for(const auto& obst : *obstacles)
            {
                // std::cout<<obst->getObstacleType()<<std::endl;
                if (obst->getObstacleType() == Obstacle::Type::POINT)
                {
                    // visualize point obstacle
                    Eigen::Vector2d pos = boost::static_pointer_cast<PointObstacle>(obst)->position();
                    // std::cout<<pos.x()<<" "<<pos.y()<<std::endl;
                    cv::Point pc = scalePoint(pos.x(), pos.y(), scale, offset_x, offset_y);
                    cv::circle(map, pc, 5, cv::Scalar(0,0,0), -1);
                }
                if (obst->getObstacleType() == Obstacle::Type::LINE)
                {
                    // visualize line obstacle
                    Eigen::Vector2d start = boost::static_pointer_cast<LineObstacle>(obst)->start();
                    Eigen::Vector2d end = boost::static_pointer_cast<LineObstacle>(obst)->end();
                    cv::Point ps = scalePoint(start.x(), start.y(), scale, offset_x, offset_y);
                    cv::Point pe = scalePoint(end.x(), end.y(), scale, offset_x, offset_y);
                    cv::line(map, ps, pe, cv::Scalar(0,0,0), 2);
                }
                
                if (obst->getObstacleType() == Obstacle::Type::CIRCLE)
                {
                    // visualize circle obstacle
                    Eigen::Vector2d pos = boost::static_pointer_cast<CircularObstacle>(obst)->position();
                    double radius = boost::static_pointer_cast<CircularObstacle>(obst)->radius();
                    cv::Point pc = scalePoint(pos.x(), pos.y(), scale, offset_x, offset_y);
                    cv::circle(map, pc, radius*scale, cv::Scalar(0,0,0), 2);
                }


                if (obst->getObstacleType() == Obstacle::Type::POLYGON)
                {
                    // visualize polygon obstacle
                    const Point2dContainer& vertices = boost::static_pointer_cast<PolygonObstacle>(obst)->vertices();
                    for (int i=0; i<vertices.size()-1; ++i)
                    {
                        cv::Point ps = scalePoint(vertices[i].x(), vertices[i].y(), scale, offset_x, offset_y);
                        cv::Point pe = scalePoint(vertices[i+1].x(), vertices[i+1].y(), scale, offset_x, offset_y);
                        cv::line(map, ps, pe, cv::Scalar(0,0,0), 2);
                    }
                    cv::line(map, scalePoint(vertices.back().x(), vertices.back().y(), scale, offset_x, offset_y), scalePoint(vertices.front().x(), vertices.front().y(), scale, offset_x, offset_y), cv::Scalar(0,0,0), 2);
                }
            }
        }
}


private:
    LaserScanData laser_scan;
    LaserPointCloudCluster point_cloud_clusters;
    double angular_resolution; // Angular resolution in radians
    const double extra_distance = 0.5;
    std::vector<std::vector<Eigen::Vector2d>> rectangles;
    ObstContainer *obstacles;
    ObstContainer *all_points;
    double scale = 50.0;  
    double offset_x = 700; 
    double offset_y = 500;
};

class LaserScanSubscriber : public rclcpp::Node
{
public:
    LaserScanSubscriber()
        : Node("laser_scan_subscriber"), map(1500, 1500, CV_8UC3, cv::Scalar(255,255,255))
    {
        subscription_ = this->create_subscription<sensor_msgs::msg::LaserScan>(
            "scan", 10, std::bind(&LaserScanSubscriber::scanCallback, this, std::placeholders::_1));
        odom_sub_ = this->create_subscription<nav_msgs::msg::Odometry>(
            "odom", 10, std::bind(&LaserScanSubscriber::odomCallback, this, std::placeholders::_1));      
    }
    void scanCallback(const sensor_msgs::msg::LaserScan::SharedPtr msg)
    {
        RCLCPP_INFO(this->get_logger(), "Received LaserScan data: range size: %zu", msg->ranges.size());
        LaserScanData scan;
        map = cv::Scalar(255, 255, 255); 
        for (size_t i = 0; i < msg->ranges.size(); ++i)
        {
            // ensuring valid value (not nan, inf)
            if (std::isnan(msg->ranges[i]) || std::isinf(msg->ranges[i]))
                continue;
            double range = msg->ranges[i];
            double angle = msg->angle_min + i * msg->angle_increment;
            Eigen::Vector2d point = transformToGlobal(range, angle);
            scan.emplace_back(range, angle, point);
        }
        TestClustering clustering(scan);
        clustering.clusterPointCloud();
        clustering.printClusters();
        clustering.rectangleFitting();
        clustering.obsVisualize(map, false, robot_pose);
        cv::imshow("Map", map);
        cv::waitKey(1);


    }
    Eigen::Vector2d transformToGlobal(double r, double theta)
    {
        // from r and theta to x,y
        Eigen::Vector2d pose;
        pose.x() = robot_pose.x + r * cos(robot_pose.theta + theta);
        pose.y() = robot_pose.y + r * sin(robot_pose.theta + theta);
        return pose;
    }
    void odomCallback(const nav_msgs::msg::Odometry::SharedPtr msg)
    {
        robot_pose.x = msg->pose.pose.position.x;
        robot_pose.y = msg->pose.pose.position.y;
        robot_pose.theta = tf2::getYaw(msg->pose.pose.orientation);

    }
    rclcpp::Subscription<sensor_msgs::msg::LaserScan>::SharedPtr subscription_;
    rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr odom_sub_;
    RobotPose robot_pose;
    cv::Mat map;
};

int main(int argc, char **argv)
{
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<LaserScanSubscriber>());
    rclcpp::shutdown();
    return 0;
}
