# 计算机图形学

*Created by KennyS*

---

## 二维图形变换、实现

### 平移变换

1. 推导
平移变换是将图形在二维平面上按照某个向量$(\Delta x, \Delta y)$进行移动。对于平面上任意一点$(x,y)$，平移后的坐标$({x}', {y}')$计算公式为：
$${x}' = x + \Delta x \\ 
{y}' = y + \Delta y
$$

2. C++代码
```cpp
#include <iostream>
#include <vector>

struct Point {
    double x, y;
};

void translate(std::vector<Point>& points, double dx, double dy) {
    for (auto& p : points) {
        p.x += dx;
        p.y += dy;
    }
}

int main() {
    std::vector<Point> polygon = {{1, 2}, {3, 4}, {5, 1}};
    double dx = 2, dy = 3;
    translate(polygon, dx, dy);
    for (const auto& p : polygon) {
        std::cout << "Translated Point: (" << p.x << ", " << p.y << ")" << std::endl;
    }
    return 0;
}
```

3. python代码
```python
import matplotlib.pyplot as plt

def translate(points, dx, dy):
    return [(x + dx, y + dy) for x, y in points]

points = [(1, 2), (3, 4), (5, 1)]
dx, dy = 2, 3
translated_points = translate(points, dx, dy)

plt.figure()
plt.plot(*zip(*points), 'bo-', label='Original')
plt.plot(*zip(*translated_points), 'ro-', label='Translated')
plt.axis("equal")
plt.legend()
plt.show()
```

### 比例变换

1. 推导
比例变换是改变图形大小的操作。设$k_{x}$和$k_{y}$分别是沿着$x$轴和$y$轴的缩放因子。则$(x,y)$缩放后的坐标$({x}', {y}')$可以通过如下方式计算
$$
{x}' = k_{x} \cdot x \\
{y}' = k_{y} \cdot y
$$

2. C++代码
```cpp
#include <iostream>
#include <vector>

struct Point {
    double x, y;
};

void scale(std::vector<Point>& points, double kx, double ky) {
    for (auto& p : points) {
        p.x *= kx;
        p.y *= ky;
    }
}

int main() {
    std::vector<Point> polygon = {{1, 2}, {3, 4}, {5, 1}};
    double kx = 2, ky = 3;
    scale(polygon, kx, ky);
    for (const auto& p : polygon) {
        std::cout << "Scaled Point: (" << p.x << ", " << p.y << ")" << std::endl;
    }
    return 0;
}
```

3. python代码
```python
import matplotlib.pyplot as plt

def scale(points, kx, ky):
    return [(x * kx, y * ky) for x, y in points]

points = [(1, 2), (3, 4), (5, 1)]
kx, ky = 2, 3
scaled_points = scale(points, kx, ky)

plt.figure()
plt.plot(*zip(*points), 'bo-', label='Original')
plt.plot(*zip(*scaled_points), 'ro-', label='Scaled')
plt.axis("equal")
plt.legend()
plt.show()
```

