# vtk_extraction.pyx

import vtk
from scipy.spatial import ConvexHull
import numpy as np
from sklearn.cluster import KMeans
#from libc.stdlib cimport malloc, free


# 读取VTK文件
def read_vtk(file_name):
    reader = vtk.vtkPolyDataReader()
    reader.SetFileName(file_name)
    reader.Update()
    return reader.GetOutput()

# 获取多面体的点
def get_points(poly_data):
    cdef list points = []
    for i in range(poly_data.GetNumberOfPoints()):
        points.append(poly_data.GetPoint(i))
    return points

# 计算凸包并获取顶点
def compute_convex_hull(points):
    hull = ConvexHull(points)
    vertices = hull.vertices
    hull_points = [points[vertex] for vertex in vertices]
    return hull_points, hull

# 聚类并选择代表性特征点
def select_representative_points(points, int num_points):
    kmeans = KMeans(n_clusters=num_points)
    kmeans.fit(points)
    centers = kmeans.cluster_centers_
    return centers



def read_vtk_and_extract_points(input_file, int n):
    poly_data = read_vtk(input_file)
    points = get_points(poly_data)
    
    
    hull_points, hull = compute_convex_hull(points)
    num_points = min(len(hull_points), n)
    representative_points = select_representative_points(hull_points, num_points)
    
    #print(f"Points output: {representative_points}", flush=True)
    #print("*--------------------------------------------*")
    #print(f"Points length is {len(representative_points)}...")

    return representative_points.tolist()
