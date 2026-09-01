#pragma once

// ROS 2 Humble exposed planner exceptions through exceptions.hpp. Nav2 Jazzy
// split them into planner_exceptions.hpp. This build-only shim keeps the public
// controller source and policy logic unmodified.
#include "nav2_core/planner_exceptions.hpp"
