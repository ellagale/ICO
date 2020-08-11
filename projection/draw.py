
###############################################################################
#    drawing functions
###############################################################################

def draw_molecule_in_ico(molecule, ico=1, mesh_file='', scale_by=5, rotate_by=[], translate_by=[], angle1=0, angle2=90):
    """Function to draw molecule in an icosahdron or icosphere
    molecule is an molecule object of RD molecule, coords, atomlist
    ico is a setting for which icospehere you want
    mesh_file is a mesh_file to override the ico setting (set ico to 0)
    scale is the size to scale icosphere by (it's a unit ico)"""
    if ico==1:
        mesh_file="C:\\Users\\ella_\\Documents\\GitHub\\icosahedron_projection\\icosahedrons\\icosohedron.stl"
    elif ico==2:
        mesh_file="C:\\Users\\ella_\\Documents\\GitHub\\icosahedron_projection\\icosahedrons\\icosphere2.stl"
    else:
        mesh_file=mesh_file
    coords=molecule[1]
    atom_list=molecule[2]
    colours=get_colour_list(atom_list)
    figure = pyplot.figure()
    axes = mplot3d.Axes3D(figure)

    # Load the STL files and add the vectors to the plot
    your_mesh = mesh.Mesh.from_file(mesh_file)
    #axes.add_collection3d(mplot3d.art3d.Poly3DCollection(your_mesh.vectors))
    axes.add_collection3d(mplot3d.art3d.Line3DCollection(your_mesh.vectors*scale_by))
    #axes.add_collection3d(mplot3d.art3d.Patch3DCollection(coords))
    axes.scatter(coords[:,0],coords[:,1],coords[:,2],c=colours)
    normal=your_mesh.normals[-1]*5
    axes.scatter(normal[0],normal[1],normal[2],c='g',marker='$5$',s=80)
    #triangle=your_mesh.vectors[-1]
    #axes.scatter(triangle[:,0]*5,triangle[:,1]*5,triangle[:,2]*5,c='k')
    #axes.plot([0,coords[0,0]*3],[0,coords[0,1]*3],[0,coords[0,2]*3],c='k')
    # Auto scale to the mesh size
    #scale = your_mesh.points.flatten(-1)
    scale = your_mesh.points.flatten('K')*scale_by
    axes.auto_scale_xyz(scale*1.25, scale, scale)
    axes.view_init(angle1,angle2)
    pyplot.draw()
    return


def animate_molecule_in_ico(molecule,
                            molecule_name='molecule',
                            ico=1, mesh_file='',
                            scale_by=5,
                            rotate_by=[],
                            translate_by=[],
                            angle=0,
                            top=0):
    """Function to draw molecule in an icosahdron or icosphere
    molecule is an molecule object of RD molecule, coords, atomlist
    ico is a setting for which icospehere you want
    mesh_file is a mesh_file to override the ico setting (set ico to 0)
    scale is the size to scale icosphere by (it's a unit ico)"""
    if ico == 1:
        mesh_file = "C:\\Users\\ella_\\Documents\\GitHub\\icosahedron_projection\\icosahedrons\\icosohedron.stl"
    elif ico == 2:
        mesh_file = "C:\\Users\\ella_\\Documents\\GitHub\\icosahedron_projection\\icosahedrons\\icosphere2.stl"
    else:
        mesh_file = mesh_file
    coords = molecule[1]
    atom_list = molecule[2]
    colour_list = get_colour_list(atom_list)

    # print(colour_list);
    @gif.frame
    def plot_icosahedron_and_molecule(i):
        figure = pyplot.figure()
        axes = mplot3d.Axes3D(figure)

        # Load the STL files and add the vectors to the plot
        your_mesh = mesh.Mesh.from_file(mesh_file)
        # axes.add_collection3d(mplot3d.art3d.Poly3DCollection(your_mesh.vectors))
        axes.add_collection3d(mplot3d.art3d.Line3DCollection(your_mesh.vectors * scale_by))
        # axes.add_collection3d(mplot3d.art3d.Patch3DCollection(coords))
        axes.scatter(coords[:, 0], coords[:, 1], coords[:, 2], c=colour_list)
        # for atom in range(len(atom_list)):
        #    normal=your_mesh.normals[triangles[atom]]
        #    axes.scatter(normal[0]*scale_by,normal[1]*scale_by,normal[2]*scale_by,c=colour_list[atom])
        # triangle=your_mesh.vectors[atom]
        # axes.scatter(triangle[:,0]*5,triangle[:,1]*5,triangle[:,2]*5,c=colour_list[atom])
        # axes.plot([0,coords[atom,0]*2],[0,coords[atom,1]*2],[0,coords[atom,2]*2],c=colour_list[atom])
        # Auto scale to the mesh size
        # scale = your_mesh.points.flatten(-1)
        scale = your_mesh.points.flatten('K') * scale_by
        if top == 1:
            axes.auto_scale_xyz(scale * 1.25, scale, scale)
            # axes.axis('equal')
            axes.view_init(i, angle)
        else:
            axes.auto_scale_xyz(scale, scale, scale * 0.75)
            # axes.axis('equal')
            axes.view_init(angle, i)
        axes.axis('off')

        pyplot.draw()

    frames = []
    for i in range(360):
        frame = plot_icosahedron_and_molecule(i)
        frames.append(frame)

    gif.save(frames, molecule_name + '.gif', duration=50)
    return


def animate_atom_projection_onto_ico(molecule,
                                     triangles,
                                     molecule_name='molecule',
                                     ico=1,
                                     mesh_file='',
                                     scale_by=5,
                                     rotate_by=[],
                                     translate_by=[],
                                     angle=0,
                                     top=0):
    """Function to draw molecule in an icosahdron or icosphere
    molecule is an molecule object of RD molecule, coords, atomlist
    ico is a setting for which icospehere you want
    mesh_file is a mesh_file to override the ico setting (set ico to 0)
    scale is the size to scale icosphere by (it's a unit ico)"""
    if ico == 1:
        mesh_file = "C:\\Users\\ella_\\Documents\\GitHub\\icosahedron_projection\\icosahedrons\\icosohedron.stl"
    elif ico == 2:
        mesh_file = "C:\\Users\\ella_\\Documents\\GitHub\\icosahedron_projection\\icosahedrons\\icosphere2.stl"
    else:
        mesh_file = mesh_file
    coords = molecule[1]
    atom_list = molecule[2]
    colour_list = get_colour_list(atom_list)

    # print(colour_list);
    @gif.frame
    def plot_icosahedron_and_molecule_and_projection(i):
        figure = pyplot.figure()
        axes = mplot3d.Axes3D(figure)

        # Load the STL files and add the vectors to the plot
        your_mesh = mesh.Mesh.from_file(mesh_file)
        # axes.add_collection3d(mplot3d.art3d.Poly3DCollection(your_mesh.vectors))
        axes.add_collection3d(mplot3d.art3d.Line3DCollection(your_mesh.vectors * scale_by))
        # axes.add_collection3d(mplot3d.art3d.Patch3DCollection(coords))
        axes.scatter(coords[:, 0], coords[:, 1], coords[:, 2], c=colour_list)
        for atom in range(len(atom_list)):
            normal = your_mesh.normals[triangles[atom]]
            axes.scatter(normal[0] * scale_by, normal[1] * scale_by, normal[2] * scale_by, c=colour_list[atom])
            triangle = your_mesh.vectors[triangles[atom]]
            axes.scatter(triangle[:, 0] * scale_by, triangle[:, 1] * scale_by, triangle[:, 2] * scale_by,
                         c=colour_list[atom])
            axes.plot([0, coords[atom, 0] * scale_by], [0, coords[atom, 1] * scale_by], [0, coords[atom, 2] * scale_by],
                      c=colour_list[atom])
        # Auto scale to the mesh size
        # scale = your_mesh.points.flatten(-1)
        scale = your_mesh.points.flatten('K') * scale_by
        if top == 1:
            axes.auto_scale_xyz(scale, scale, scale)
            # axes.axis('equal')
            axes.view_init(i, angle)
        else:
            axes.auto_scale_xyz(scale, scale, scale * 0.75)
            # axes.axis('equal')
            axes.view_init(angle, i)
            # axes.auto_scale_xyz(scale, scale, scale*0.75)
        axes.view_init(angle, i)
        axes.axis('off')
        pyplot.draw()

    frames = []
    for i in range(360):
        frame = plot_icosahedron_and_molecule_and_projection(i)
        frames.append(frame)

    gif.save(frames, molecule_name + '.gif', duration=50)
    return

