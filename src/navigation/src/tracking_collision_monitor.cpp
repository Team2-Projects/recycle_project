// Local Humble 1.1.20 adapter; geometric algorithms remain in nav2_collision_monitor.
// One replacement Monitor process, not an additional supervisor or scan relay.
#include <algorithm>
#include <chrono>
#include <cmath>
#include <memory>
#include <stdexcept>
#include <vector>
#include "nav2_collision_monitor/collision_monitor_node.hpp"
#include "nav2_util/node_utils.hpp"
#include "navigation_interface/msg/collision_command.hpp"

namespace cm = nav2_collision_monitor;
using Packet = navigation_interface::msg::CollisionCommand;
using Steady = std::chrono::steady_clock;

class CheckedScan : public cm::Scan
{
public:
  CheckedScan(const nav2_util::LifecycleNode::WeakPtr & node,
    const std::shared_ptr<tf2_ros::Buffer> tf, const std::string & base,
    const std::string & odom, double timeout, double future, double transform_tolerance)
  : Scan(node, "scan", tf, base, odom, tf2::durationFromSec(transform_tolerance),
      rclcpp::Duration::from_seconds(timeout), false), timeout_(timeout), future_(future)
  {
    configure();
    auto n = node_.lock();
    // Replace Scan's own callback, not a second independently observed stream.
    data_sub_ = n->create_subscription<sensor_msgs::msg::LaserScan>(
      n->get_parameter("scan.topic").as_string(), rclcpp::SensorDataQoS().keep_last(1),
      [this](sensor_msgs::msg::LaserScan::ConstSharedPtr msg) { receive(msg); });
  }

  bool read(std::vector<cm::Point> & points)
  {
    const auto now = node_.lock()->now();
    if (!getEnabled() || !fresh(now)) { return false; }
    Scan::getData(now, points);
    // Every accepted scan has at least one finite in-range beam. Therefore
    // empty output here means projection/TF failed, never "clear space".
    return !points.empty();
  }

private:
  bool fresh(const rclcpp::Time & now) const
  {
    if (!data_) { return false; }
    const auto age = (now - rclcpp::Time(data_->header.stamp)).seconds();
    return std::chrono::duration<double>(Steady::now() - received_).count() <= timeout_ &&
           age >= -future_ && age <= timeout_;
  }

  void receive(sensor_msgs::msg::LaserScan::ConstSharedPtr msg)
  {
    if (msg->header.stamp.sec < 0 || msg->header.stamp.nanosec >= 1000000000u) { return; }
    const auto now = node_.lock()->now();
    const auto stamp = rclcpp::Time(msg->header.stamp);
    const double age = (now - stamp).seconds();
    const bool finite = std::isfinite(msg->angle_min) && std::isfinite(msg->angle_max) &&
      std::isfinite(msg->angle_increment) && std::isfinite(msg->range_min) && std::isfinite(msg->range_max);
    if (stamp.nanoseconds() <= 0 || age < -future_ || age > timeout_ ||
      msg->header.frame_id.empty() || !finite || msg->ranges.size() < 2 ||
      msg->angle_increment <= 0 || msg->range_max <= msg->range_min ||
      !std::any_of(msg->ranges.begin(), msg->ranges.end(), [&](float r) {
        return std::isfinite(r) && r >= std::max(msg->range_min, .001f) && r <= msg->range_max;
      })) { return; }
    if (data_) {
      const auto previous = rclcpp::Time(data_->header.stamp);
      // A replay never extends validity. A fresh new clock/frame epoch may
      // restart after a real outage; there is no cross-node recovery FSM.
      if (stamp == previous || (fresh(now) &&
        (stamp < previous || msg->header.frame_id != data_->header.frame_id))) { return; }
    }
    data_ = msg;
    received_ = Steady::now();
  }
  double timeout_, future_;
  Steady::time_point received_;
};

class FixedFootprint : public cm::Polygon
{
public:
  using Polygon::Polygon;
  bool configure()
  {
    if (!Polygon::configure()) { return false; }
    footprint_sub_.reset();
    auto node = node_.lock();
    nav2_util::declare_parameter_if_not_declared(
      node, polygon_name_ + ".points", rclcpp::ParameterValue(std::vector<double>{}));
    const auto values = node->get_parameter(polygon_name_ + ".points").as_double_array();
    if (values.size() < 6 || values.size() % 2 ||
      !std::all_of(values.begin(), values.end(), [](double v) {return std::isfinite(v);})) { return false; }
    poly_.clear();
    for (size_t i = 0; i < values.size(); i += 2) { poly_.push_back({values[i], values[i + 1]}); }
    return true;
  }
};

class TrackingCollisionMonitor : public cm::CollisionMonitor
{
protected:
  nav2_util::CallbackReturn on_configure(const rclcpp_lifecycle::State & state) override
  {
    if (CollisionMonitor::on_configure(state) != nav2_util::CallbackReturn::SUCCESS) {
      return nav2_util::CallbackReturn::FAILURE;
    }
    // No uncorrelated Twist path remains after configuration.
    cmd_vel_in_sub_.reset();
    cmd_vel_out_pub_.reset();
    sources_.clear();
    auto node = shared_from_this();
    nav2_util::declare_parameter_if_not_declared(node, "scan_future_tolerance_sec", rclcpp::ParameterValue(.1));
    const auto base = get_parameter("base_frame_id").as_string();
    const double timeout = get_parameter("source_timeout").as_double();
    const double future = get_parameter("scan_future_tolerance_sec").as_double();
    if (!std::isfinite(timeout) || timeout <= 0 || !std::isfinite(future) || future < 0 || polygons_.empty()) {
      return nav2_util::CallbackReturn::FAILURE;
    }
    scan_ = std::make_shared<CheckedScan>(node, tf_buffer_, base,
      get_parameter("odom_frame_id").as_string(), timeout, future,
      get_parameter("transform_tolerance").as_double());
    for (auto & polygon : polygons_) {
      if (polygon->getActionType() == cm::APPROACH) {
        auto fixed = std::make_shared<FixedFootprint>(node, polygon->getName(), tf_buffer_, base,
          tf2::durationFromSec(0.));
        polygon.reset();
        if (!fixed->configure()) { return nav2_util::CallbackReturn::FAILURE; }
        polygon = fixed;
      }
      if (!polygon->getEnabled()) { return nav2_util::CallbackReturn::FAILURE; }
    }
    pub_ = create_publisher<Packet>(get_parameter("cmd_vel_out_topic").as_string(), 1);
    sub_ = create_subscription<Packet>(get_parameter("cmd_vel_in_topic").as_string(), 1,
      [this](Packet::ConstSharedPtr raw) { process_checked(*raw); });
    // Geometry and source configuration are immutable for this lifecycle.
    parameters_guard_ = add_on_set_parameters_callback([](const std::vector<rclcpp::Parameter> & params) {
      rcl_interfaces::msg::SetParametersResult result;
      result.successful = std::all_of(params.begin(), params.end(), [](const auto & p) {
        return p.get_name() == "use_sim_time";
      });
      result.reason = "Restart Monitor to change collision configuration";
      return result;
    });
    return nav2_util::CallbackReturn::SUCCESS;
  }

  nav2_util::CallbackReturn on_activate(const rclcpp_lifecycle::State &) override
  {
    pub_->on_activate();
    for (auto & polygon : polygons_) { polygon->activate(); }
    process_active_ = true;
    createBond();
    return nav2_util::CallbackReturn::SUCCESS;
  }
  nav2_util::CallbackReturn on_deactivate(const rclcpp_lifecycle::State &) override
  {
    process_active_ = false;
    pub_->on_deactivate();
    destroyBond();
    for (auto & polygon : polygons_) { polygon->deactivate(); }
    return nav2_util::CallbackReturn::SUCCESS;
  }
  nav2_util::CallbackReturn on_cleanup(const rclcpp_lifecycle::State & state) override
  {
    sub_.reset(); pub_.reset(); scan_.reset(); parameters_guard_.reset();
    return CollisionMonitor::on_cleanup(state);
  }

private:
  void process_checked(const Packet & raw)
  {
    if (!process_active_) { return; }
    Packet safe;
    safe.request_id = raw.request_id;
    std::vector<cm::Point> points;
    safe.input_valid = nav2_util::validateTwist(raw.velocity) &&
      raw.velocity.linear.y == 0 && raw.velocity.linear.z == 0 &&
      raw.velocity.angular.x == 0 && raw.velocity.angular.y == 0 && scan_->read(points);
    if (safe.input_valid) {
      const cm::Velocity velocity{raw.velocity.linear.x, 0., raw.velocity.angular.z};
      cm::Action action{cm::DO_NOTHING, velocity};
      for (const auto & polygon : polygons_) {
        if (action.action_type == cm::STOP) { break; }
        if (polygon->getActionType() == cm::APPROACH) {
          processApproach(polygon, points, velocity, action);
        } else {
          processStopSlowdown(polygon, points, velocity, action);
        }
      }
      safe.velocity.linear.x = action.req_vel.x;
      safe.velocity.angular.z = action.req_vel.tw;
    }
    // Always answer, including repeated zero and unusable input. This removes
    // Humble's stop_pub_timeout ambiguity without an idle heartbeat protocol.
    pub_->publish(safe);
  }
  std::shared_ptr<CheckedScan> scan_;
  rclcpp::Subscription<Packet>::SharedPtr sub_;
  rclcpp_lifecycle::LifecyclePublisher<Packet>::SharedPtr pub_;
  rclcpp::node_interfaces::OnSetParametersCallbackHandle::SharedPtr parameters_guard_;
};

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  // Single executor: scan adoption and processing use the same actual sample.
  auto monitor = std::make_shared<TrackingCollisionMonitor>();
  rclcpp::spin(monitor->get_node_base_interface());
  rclcpp::shutdown();
  return 0;
}
