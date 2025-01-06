#include <iostream>
#include <Eigen/Dense>
#include <vector>
#include <cmath>
#include <utility>
#include"obstacles.h"
using namespace teb_local_planner;


double findDifferenceOrientation(double angle1, double angle2)
{
    double diff = angle1 - angle2;
    while (diff > M_PI)
        diff -= 2 * M_PI;
    while (diff < -M_PI)
        diff += 2 * M_PI;
    return diff;
}

class TestClustering
{
public:
    TestClustering(const LaserScanData& laser_scan_)
        : laser_scan(laser_scan_), angular_resolution(M_PI / 180.0) // Example angular resolution: 1 degree
    {
        obstacles = new ObstContainer();
    }

    ~TestClustering()
    {
        delete obstacles;
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
                    q = closenessCriterion(C1, C2, 0.0001) + areaCriterion(C1, C2);
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

        Eigen::VectorXd C1_max = c1_max - C1.array(); 
        Eigen::VectorXd C1_min = C1.array() - c1_min;
        Eigen::VectorXd D1, D2;
        if(C1_max.squaredNorm() < C1_min.squaredNorm()){
            D1 = C1_max;
        }
        else{
            D1 = C1_min;
        }
        Eigen::VectorXd C2_max = c2_max - C2.array(); 
        Eigen::VectorXd C2_min = C2.array() - c2_min;
        if(C2_max.squaredNorm() < C2_min.squaredNorm()){
            D2 = C2_max;
        }
        else{
            D2 = C2_min;
        }

        double d, min;
        double b = 0 ;
        for (int i = 0; i < D1.size(); ++i) 
        {
            min = std::min(D1(i),D2(i));
            d = std::max(min, d0);
            b = b + 1/d;
        }
        
        return b; 
    }

    double varianceCriterion(const Eigen::VectorXd &C1, const Eigen::VectorXd &C2)
    {
        double c1_max = C1.maxCoeff();
        double c1_min = C1.minCoeff();
        double c2_max = C2.maxCoeff();
        double c2_min = C2.minCoeff();

        Eigen::VectorXd C1_max = c1_max - C1.array(); 
        Eigen::VectorXd C1_min = C1.array() - c1_min;
        Eigen::VectorXd D1, D2;
        if(C1_max.squaredNorm() < C1_min.squaredNorm()){
            D1 = C1_max;
        }
        else{
            D1 = C1_min;
        }
        Eigen::VectorXd C2_max = c2_max - C2.array(); 
        Eigen::VectorXd C2_min = C2.array() - c2_min;
        if(C2_max.squaredNorm() < C2_min.squaredNorm()){
            D2 = C2_max;
        }
        else{
            D2 = C2_min;
        }
        Eigen::VectorXd E1(D1.size()), E2(D2.size());
        for (int i = 0; i < E1.size(); i++)
        {
            if (D1(i) < D2(i)) E1(i) = D1(i);
            if (D2(i) < D1(i)) E2(i) = D2(i);
        }
        double gamma = - sqrt((E1.array() - E1.mean()).square().sum() / (E1.size() - 1)) - sqrt((E2.array() - E2.mean()).square().sum() / (E2.size() - 1));
        return gamma;
    }


private:
    LaserScanData laser_scan;
    LaserPointCloudCluster point_cloud_clusters;
    double angular_resolution; // Angular resolution in radians
    const double extra_distance = 0.5;
    std::vector<std::vector<Eigen::Vector2d>> rectangles;
    ObstContainer *obstacles;
};

// Helper function to generate sample laser scan data
LaserScanData generateSampleLaserScan()
{
    LaserScanData scan;
    double center_x = 5.0;
    double center_y = 5.0;
    double width = 2.0;
    double height = 1.0;
    int points_per_side = 10;

    // Tạo các điểm cho bốn cạnh của hình chữ nhật
    for (int i = 0; i < points_per_side; ++i)
    {
        double ratio = static_cast<double>(i) / (points_per_side - 1);
        // Cạnh trên
        double x = center_x - width / 2 + ratio * width;
        double y = center_y + height / 2;
        double range = sqrt(x * x + y * y);
        double angle = atan2(y, x);
        Eigen::Vector2d point(x, y);
        scan.emplace_back(range, angle, point);

        // Cạnh dưới
        x = center_x - width / 2 + ratio * width;
        y = center_y - height / 2;
        range = sqrt(x * x + y * y);
        angle = atan2(y, x);
        point = Eigen::Vector2d(x, y);
        scan.emplace_back(range, angle, point);
    }


    // Generate a cluster of points in a straight line
    for (int i = 0; i < 10; ++i)
    {
        double angle = M_PI / 4; // 45 degrees
        double range = 5.0 + i * 0.1; // Increasing range
        Eigen::Vector2d point(range * std::cos(angle), range * std::sin(angle));
        scan.emplace_back(range, angle, point);
    }

    // Gap between clusters
    for (int i = 0; i < 5; ++i)
    {
        double angle = M_PI / 3; // 60 degrees
        double range = 10.0 + i * 0.1;
        Eigen::Vector2d point(range * std::cos(angle), range * std::sin(angle));
        scan.emplace_back(range, angle, point);
    }

    // Another cluster with different angle
    for (int i = 0; i < 8; ++i)
    {
        double angle = M_PI / 6; // 30 degrees
        double range = 7.0 + i * 0.2;
        Eigen::Vector2d point(range * std::cos(angle), range * std::sin(angle));
        scan.emplace_back(range, angle, point);
    }

    return scan;
}

int main()
{
    LaserScanData laser_scan = generateSampleLaserScan();
    TestClustering clustering(laser_scan);
    clustering.clusterPointCloud();
    clustering.printClusters();
    clustering.rectangleFitting();

    return 0;
}