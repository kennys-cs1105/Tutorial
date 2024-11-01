
import vtk
from scipy.spatial import ConvexHull
import numpy as np
from sklearn.cluster import KMeans
import time



# 读取VTK文件
def read_vtk(file_name):
    reader = vtk.vtkPolyDataReader()
    reader.SetFileName(file_name)
    reader.Update()
    return reader.GetOutput()

# 获取多面体的点
def get_points(poly_data):
    points = []
    for i in range(poly_data.GetNumberOfPoints()):
        points.append(poly_data.GetPoint(i))
    return points

# 计算凸包并获取顶点
def compute_convex_hull(points):
    hull = ConvexHull(points)
    vertices = hull.vertices
    hull_points = [points[vertex] for vertex in vertices]
    return hull_points,hull

# 聚类并选择代表性特征点
def select_representative_points(points, num_points):
    kmeans = KMeans(n_clusters=num_points)
    kmeans.fit(points)
    centers = kmeans.cluster_centers_
    return centers



def read_vtk_and_extract_points(input_file, n):
    
    start_time = time.time()
    poly_data = read_vtk(input_file)  # 替换为实际的VTK文件路径  "D:/workspace/2024/niu/wedge.vtk"
    end_time = time.time()
    print(f"读取 VTK 文件所需时间: {end_time - start_time} 秒")

    start_time = time.time()
    points = get_points(poly_data)
    end_time = time.time()
    print(f"获取多面体的点所需时间: {end_time - start_time} 秒")

    
    start_time = time.time()
    hull_points, hull = compute_convex_hull(points)
    end_time = time.time()
    print(f"计算凸包并获取顶点所需时间: {end_time - start_time} 秒")
    
    num_points = min(len(hull_points), n)

    start_time = time.time()
    representative_points = select_representative_points(hull_points, num_points).tolist()
    end_time = time.time()
    print(f"聚类并选择代表性特征点所需时间: {end_time - start_time} 秒")
 
    print(type(representative_points))
    return representative_points
    # return representative_points.flatten()


if __name__ == "__main__":
    input_file = "/data/renjuan/workspace/fit_polyhedra/wedge/wedge.vtk"  # 替换为实际的VTK文件路径
    x = read_vtk_and_extract_points(input_file, 6)
    print(x)            
    
  
