/*********************************************************************
 *
 * Software License Agreement (BSD License)
 *
 *  Copyright (c) 2016,
 *  TU Dortmund - Institute of Control Theory and Systems Engineering.
 *  All rights reserved.
 *
 *  Redistribution and use in source and binary forms, with or without
 *  modification, are permitted provided that the following conditions
 *  are met:
 *
 *   * Redistributions of source code must retain the above copyright
 *     notice, this list of conditions and the following disclaimer.
 *   * Redistributions in binary form must reproduce the above
 *     copyright notice, this list of conditions and the following
 *     disclaimer in the documentation and/or other materials provided
 *     with the distribution.
 *   * Neither the name of the institute nor the names of its
 *     contributors may be used to endorse or promote products derived
 *     from this software without specific prior written permission.
 *
 *  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 *  "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
 *  LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
 *  FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
 *  COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
 *  INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
 *  BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 *  LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 *  CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
 *  LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
 *  ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 *  POSSIBILITY OF SUCH DAMAGE.
 *
 * Author: Christoph Rösmann
 *********************************************************************/

#include "obstacles.h"
// #include <teb_local_planner/misc.h>

namespace teb_local_planner
{
  void PolygonObstacle::visualize(cv::Mat &map, const cv::Scalar &color, double gain_x, double gain_y, double map_height) const
  {
    std::vector<cv::Point> polygon;
    for (int i = 0; i < this->vertices_.size(); i++)
    {
      Point2DPixel point(this->vertices_[i].x(), this->vertices_[i].y(), gain_x, gain_y, map_height);
      polygon.push_back(cv::Point(point.x, point.y));
    }
    cv::fillConvexPoly(map, polygon, color);
  }

void PolygonObstacle::fixPolygonClosure()
{
  if (vertices_.size()<2)
    return;
  
  if (vertices_.front().isApprox(vertices_.back()))
    vertices_.pop_back();
}

void PolygonObstacle::calcCentroid()
{
  if (vertices_.empty())
  {
    centroid_.setConstant(NAN);
    std::cout << "PolygonObstacle::calcCentroid(): number of vertices is empty. the resulting centroid is a vector of NANs." << std::endl;
    return;
  }
  
  // if polygon is a point
  if (noVertices()==1)
  {
    centroid_ = vertices_.front();
    return;
  }
  
  // if polygon is a line:
  if (noVertices()==2)
  {
    centroid_ = 0.5*(vertices_.front() + vertices_.back());
    return;
  }
  
  // otherwise:
  
  centroid_.setZero();
    
  // calculate centroid (see wikipedia http://de.wikipedia.org/wiki/Geometrischer_Schwerpunkt#Polygon)
  double A = 0;  // A = 0.5 * sum_0_n-1 (x_i * y_{i+1} - x_{i+1} * y_i)
  for (int i=0; i < noVertices()-1; ++i)
  {
    A += vertices_.at(i).coeffRef(0) * vertices_.at(i+1).coeffRef(1) - vertices_.at(i+1).coeffRef(0) * vertices_.at(i).coeffRef(1);
  }
  A += vertices_.at(noVertices()-1).coeffRef(0) * vertices_.at(0).coeffRef(1) - vertices_.at(0).coeffRef(0) * vertices_.at(noVertices()-1).coeffRef(1);
  A *= 0.5;
  
  if (A!=0)
  {
    for (int i=0; i < noVertices()-1; ++i)
    {
      double aux = (vertices_.at(i).coeffRef(0) * vertices_.at(i+1).coeffRef(1) - vertices_.at(i+1).coeffRef(0) * vertices_.at(i).coeffRef(1));
      centroid_ +=  ( vertices_.at(i) + vertices_.at(i+1) )*aux;
    }
    double aux = (vertices_.at(noVertices()-1).coeffRef(0) * vertices_.at(0).coeffRef(1) - vertices_.at(0).coeffRef(0) * vertices_.at(noVertices()-1).coeffRef(1));
    centroid_ +=  ( vertices_.at(noVertices()-1) + vertices_.at(0) )*aux;
    centroid_ /= (6*A);	
  }
  else // A == 0 -> all points are placed on a 'perfect' line
  {
    // seach for the two outer points of the line with the maximum distance in between
    int i_cand = 0;
    int j_cand = 0;
    double max_dist = 0;
    for (int i=0; i< noVertices(); ++i)
    {
      for (int j=i+1; j< noVertices(); ++j) // start with j=i+1
      {
        double dist = (vertices_[j] - vertices_[i]).norm();
        if (dist > max_dist)
        {
          max_dist = dist;
          i_cand = i;
          j_cand = j;
        }
      }
    }
    // calc centroid of that line
    centroid_ = 0.5*(vertices_[i_cand] + vertices_[j_cand]);
  }
}



Eigen::Vector2d PolygonObstacle::getClosestPoint(const Eigen::Vector2d& position) const
{
  // the polygon is a point
  if (noVertices() == 1)
  {
    return vertices_.front();
  }
  
  if (noVertices() > 1)
  {
    
    Eigen::Vector2d new_pt = closest_point_on_line_segment_2d(position, vertices_.at(0), vertices_.at(1));
    
    if (noVertices() > 2) // real polygon and not a line
    {
      double dist = (new_pt-position).norm();
      Eigen::Vector2d closest_pt = new_pt;
      
      // check each polygon edge
      for (int i=1; i<noVertices()-1; ++i) // skip the first one, since we already checked it (new_pt)
      {
        new_pt = closest_point_on_line_segment_2d(position, vertices_.at(i), vertices_.at(i+1));
        double new_dist = (new_pt-position).norm();
        if (new_dist < dist)
        {
          dist = new_dist;
          closest_pt = new_pt;
        }
      }
      // afterwards we check the edge between goal and start (close polygon)
      new_pt = closest_point_on_line_segment_2d(position, vertices_.back(), vertices_.front());
      double new_dist = (new_pt-position).norm();
      if (new_dist < dist)
        return new_pt;
      else
        return closest_pt;
    }
    else
    {
      return new_pt; // closest point on line segment
    }
  }

  std::cout<< "PolygonObstacle::getClosestPoint() cannot find any closest point. Polygon ill-defined?" << std::endl;
  return Eigen::Vector2d::Zero(); // todo: maybe boost::optional?
}


bool PolygonObstacle::checkLineIntersection(const Eigen::Vector2d& line_start, const Eigen::Vector2d& line_end, double min_dist) const
{
  // Simple strategy, check all edge-line intersections until an intersection is found...
  // check each polygon edge
  for (int i=0; i<noVertices()-1; ++i)
  {
    if ( check_line_segments_intersection_2d(line_start, line_end, vertices_.at(i), vertices_.at(i+1)) ) 
      return true;
  }
  if (noVertices()==2) // if polygon is a line
    return false;
  
  return check_line_segments_intersection_2d(line_start, line_end, vertices_.back(), vertices_.front()); //otherwise close polygon
}

void PointObstacle::visualize(cv::Mat &map, const cv::Scalar &color, double gain_x, double gain_y, double map_height) const
{
  Point2DPixel point(this->pos_.x(), this->pos_.y(), gain_x, gain_y, map_height);
    cv::circle(map, cv::Point(point.x, point.y), 2, color, -1);
}



void CircularObstacle::visualize(cv::Mat &map, const cv::Scalar &color, double gain_x, double gain_y, double map_height) const
{
  Point2DPixel center(this->pos_.x(), this->pos_.y(), gain_x, gain_y, map_height);
  int radius = int(this->radius_ * meter_to_pixel);
  cv::circle(map, cv::Point(center.x, center.y), radius, color, -1);
}

void LineObstacle::visualize(cv::Mat &map, const cv::Scalar &color, double gain_x, double gain_y, double map_height) const
{
  Point2DPixel point1(this->start_.x(), this->start_.y(), gain_x, gain_y, map_height);
  Point2DPixel point2(this->end_.x(), this->end_.y(), gain_x, gain_y, map_height);
  cv::line(map, cv::Point(point1.x, point1.y), cv::Point(point2.x, point2.y), color, int(0.1*meter_to_pixel));
}

void PillObstacle::visualize(cv::Mat &map, const cv::Scalar &color, double gain_x, double gain_y, double map_height) const
{
  Point2DPixel point1(this->start_.x(), this->start_.y(), gain_x, gain_y, map_height);
  Point2DPixel point2(this->end_.x(), this->end_.y(), gain_x, gain_y, map_height);
  int radius = int(this->radius_ * meter_to_pixel);
  cv::line(map, cv::Point(point1.x, point1.y), cv::Point(point2.x, point2.y), color, radius*2);
  cv::circle(map, cv::Point(point1.x, point1.y), radius, color, -1);
  cv::circle(map, cv::Point(point2.x, point2.y), radius, color, -1);
}

// int PointObstacle::type() const
// {
//   return 1;
// }
// int LineObstacle::type() const
// {
//   return 2;
// }
// int CircularObstacle::type() const
// {
//   return 3;
// }
// int PolygonObstacle::type() const
// {
//   return 4;
// }
// int PillObstacle::type() const
// {
//   return 5;
// }

} // namespace teb_local_planner
