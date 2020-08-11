

#############################################################################
# Maps (projects) 3D coords of a molecule to to an icosphere
#############################################################################

def calculate_triangle_for_coords(coords, scale_by, your_mesh):
    """Returns the triangle for each atom, normal and cosine distance
    this projects the coords onto a sphere and tells you which triangle
    each molecule hits"""
    triangles=[]
    for atom in coords:
        dot_products=[]
        length_of_cross_products=[]
        cosine_list=[]
        for normal in your_mesh.normals:
            #print(normal)
            normal=normal*scale_by
            #print(np.dot(normal,atom))
            #print(scipy.spatial.distance.cosine(normal,atom))
            dot_products.append(np.dot(normal,atom))
            length_of_cross_products.append(LA.norm(np.cross(normal,atom)))
            cosine_list.append(scipy.spatial.distance.cosine(normal,atom))
            #print(LA.norm(np.cross(normal,atom)))
        triangles.append(np.where(cosine_list==min(cosine_list))[0][0])
    return triangles