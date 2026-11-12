// stencil.cpp - the stencils (all multiplied by h^2, i.e. h^2 * (-Laplace_h)).
#include "common.hpp"
#include <algorithm>
#include <cstdlib>
#include <stdexcept>

static void add(Stencil& s, int di, int dj, double c) {
  s.di.push_back(di);
  s.dj.push_back(dj);
  s.c.push_back(c);
  s.radius = std::max(s.radius, std::max(std::abs(di), std::abs(dj)));
}
// the four points at distance r on the axes
static void add_cross(Stencil& s, int r, double c) {
  add(s, -r, 0, c); add(s, r, 0, c); add(s, 0, -r, c); add(s, 0, r, c);
}

std::vector<std::string> stencil_names() { return {"5pt", "9pt", "9pt-cross", "13pt-cross"}; }

Stencil make_stencil(const std::string& name) {
  Stencil s;
  s.name = name;
  if (name == "5pt") {
    // classical 5-point stencil: [0 -1 0; -1 4 -1; 0 -1 0]
    s.sc = 4.0;
    add_cross(s, 1, -1.0);
  } else if (name == "9pt") {
    // compact 9-point stencil: 1/6 [-1 -4 -1; -4 20 -4; -1 -4 -1]
    s.sc = 20.0 / 6.0;
    add_cross(s, 1, -4.0 / 6.0);
    add(s, -1, -1, -1.0 / 6.0); add(s, -1, 1, -1.0 / 6.0);
    add(s, 1, -1, -1.0 / 6.0);  add(s, 1, 1, -1.0 / 6.0);
  } else if (name == "9pt-cross") {
    // 4th-order central differences in each direction, radius 2:
    //   1/12 [1 -16 30 -16 1] per direction  -> centre 60/12
    s.sc = 60.0 / 12.0;
    add_cross(s, 1, -16.0 / 12.0);
    add_cross(s, 2, 1.0 / 12.0);
  } else if (name == "13pt-cross") {
    // 6th-order central differences in each direction, radius 3:
    //   1/180 [-2 27 -270 490 -270 27 -2] per direction -> centre 980/180
    s.sc = 980.0 / 180.0;
    add_cross(s, 1, -270.0 / 180.0);
    add_cross(s, 2, 27.0 / 180.0);
    add_cross(s, 3, -2.0 / 180.0);
  } else {
    throw std::invalid_argument("unknown stencil '" + name + "'");
  }
  if (s.radius > MAX_RADIUS || s.np() > MAX_POINTS)
    throw std::invalid_argument("stencil too large for MAX_RADIUS/MAX_POINTS");
  return s;
}
