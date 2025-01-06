#include <iostream>
#include "homotopy_class_planner.h"
#include "obstacles.h"
#include "pose_se2.h"
#include "teb_config.h"

using namespace teb_local_planner;
inline cv::Point scalePoint(double x, double y, double scale, double offset_x, double offset_y)
{
    int px = static_cast<int>(x * scale + offset_x);
    int py = static_cast<int>(y * scale + offset_y);
    return cv::Point(px, py);
}

int main()
{
    TebConfig cfg;
    ObstContainer my_obstacles;
    my_obstacles.push_back(ObstaclePtr(new PointObstacle(2.0, 2.0))); 
    my_obstacles.push_back(ObstaclePtr(new PointObstacle(3.0, 1.0)));
    my_obstacles.push_back(ObstaclePtr(new PointObstacle(3.0, 3.0)));
    my_obstacles.push_back(ObstaclePtr(new LineObstacle(Eigen::Vector2d(1.0,2.0), Eigen::Vector2d(2.0, 1.0))));
    my_obstacles.push_back(ObstaclePtr(new CircularObstacle(7.0, 7.0, 0.5)));
    my_obstacles.push_back(ObstaclePtr(new CircularObstacle(2.0, 3.0, 0.5)));
    HomotopyClassPlanner hcp(cfg, &my_obstacles, nullptr);
    PoseSE2 start(0,0,0), goal(15,15,0);
    Twist start_vel{0.0, 0.5};
    hcp.plan(start, goal, &start_vel, false);


    cv::Mat map(1500, 1500, CV_8UC3, cv::Scalar(255,255,255));

    double scale = 50.0;  
    double offset_x = 50; 
    double offset_y = 400;

    for(const auto& obst : my_obstacles)
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

    const auto& all_tebs = hcp.getTrajectoryContainer();
    // std::cout<<all_tebs.size()<<std::endl;
    // std::cout<<hcp.getVelocityCommand(start_vel, all_tebs.size())<<std::endl;
    //print out predicted velocity command based on start_vel
    Twist velocity;
    hcp.getVelocityCommand(velocity, all_tebs.size());

    // std::cout<<"Predicted velocity command based on start_vel: "<<velocity.linear<<" "<<velocity.angular<<std::endl;
    int idx=0;
    for (auto& planner : all_tebs)
    {
        std::cout<<"Num TEBS: "<<all_tebs.size()<<std::endl;
        auto &teb = planner->teb();
        std::vector<cv::Point> path_points;
        path_points.reserve(teb.sizePoses());
        for(int p=0; p<teb.sizePoses(); ++p)
        {
            auto &pose = teb.Pose(p);
            double x = pose.x();
            double y = pose.y();
            path_points.push_back(scalePoint(x, y, scale, offset_x, offset_y));
            // extract velocity
            // std::cout<<"dt: "<<teb.TimeDiff(p)<<std::endl;
            std::cout<<"Pose:"<<x<<" "<<y<<std::endl;
            hcp.getVelocityCommand(velocity, p);
            std::cout<<"Velocity: "<<velocity.linear<<" "<<velocity.angular<<std::endl;
        }

        cv::Scalar color( (50*idx) % 255, (100*idx)%255, (150*idx)%255 );
        idx++;

        if(path_points.size()>1)
        {
            cv::polylines(map, path_points, false, color, 2);
        }
        if(!path_points.empty())
        {
            cv::circle(map, path_points.front(), 6, color, -1);
            cv::putText(map, "Start", path_points.front(), cv::FONT_HERSHEY_SIMPLEX, 0.4, color);
            cv::circle(map, path_points.back(), 6, color, -1);
            cv::putText(map, "Goal", path_points.back(), cv::FONT_HERSHEY_SIMPLEX, 0.4, color);
        }
    }

    cv::imshow("TEB Paths", map);
    cv::waitKey(0);

    return 0;
}