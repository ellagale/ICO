
##############################################################################
#    unfolds icospheres into nets
##############################################################################



class Net(object):
    """
    A net is an unfolded icosphere.
    This generates an unfolded net
    """
    def __init__(self, self.net=None,
                self.chosen_triangle_index=0,
                self.chosen_direction=0,
                self.your_mesh=your_mesh,
                self.no_of_rows=16,
                self.no_of_cols=5,
                self.verbose=False):

    def first_row_of_net(self,
                         chosen_triangle_index,
                         chosen_direction,
                         no_of_rows,
                         no_of_cols,
                         your_mesh,
                         verbose=True):
        """This creates the first row of the net
        and everything else follows from this
        chosen_triangle_index is which triangle to start from
        chosen_direction is which of the 3 vertices to use as the top most point of ([0,1,2])
        the icosphere
        :param chosen_triangle_index: which triangle ot start from
        :param chosen_direction: which direction to unfold in (there are 3)
        :param your_mesh: which icosphere to use
        :param no_of_rows:
        :param no_of_cols:
        :param verbose:
        :return:"""
        if verbose:
            print('###### Generating top row now ######')
        net = np.empty((no_of_rows, no_of_cols), dtype='int')
        net[:] = np.nan
        all_triangle_indices = [x for x in range(len(your_mesh.vectors))]
        # net=[]
        # pick a triangle
        # chosen_triangle_index=19
        chosen_triangle = your_mesh.vectors[chosen_triangle_index]
        if verbose:
            print("chosen_triangle_index is {}".format(chosen_triangle_index))
        if verbose:
            print("chosen_triangle is \n{}".format(chosen_triangle))
        # assign to net lookup table
        net[0, 0] = int(chosen_triangle_index)
        # pick a point
        chosen_point = chosen_triangle[chosen_direction]
        if verbose:
            print("chosen_point is {}".format(chosen_point))
        # find all triangles that share that point
        top_row = [x for (x, y) in enumerate(your_mesh.vectors) if np.all(np.isin(chosen_point, y))]
        if verbose:
            print("top row is {}".format(top_row))
        # find neighbours of chosen triangle
        neighbours = [top_row[y] for (y, x) in enumerate(
            your_mesh.vectors[top_row])
                      if np.count_nonzero(np.all(np.isin(x, chosen_triangle), axis=1)) == 2]
        # pick direction -- choose your next triangle
        next_triangle_index = max(neighbours)
        next_triangle = your_mesh.vectors[next_triangle_index]
        if verbose:
            print("neighbours of chosen triangle are{}".format(neighbours))
            print("next triangle in the row is {}".format(next_triangle_index))
            print("next triangle in the row is located at\n{}".format(next_triangle))
        net[0, 1] = int(next_triangle_index)
        for i in range(2, len(top_row)):
            # find neighbours of new triangle
            neighbours = [top_row[y] for (y, x) in enumerate(
                your_mesh.vectors[top_row])
                          if np.count_nonzero(np.all(np.isin(x, next_triangle), axis=1)) == 2]
            # pick the one not already in the net
            next_triangle_index = [x for x in neighbours if x not in net][0]
            net[0, i] = int(next_triangle_index)
            next_triangle = your_mesh.vectors[next_triangle_index]
        return net

    def determining_row_of_upright_triangles(self,
                                             net,
                                             row_no,
                                             no_of_cols,
                                             your_mesh,
                                             verbose=True):
        """algorithm"""
        for i in range(no_of_cols):
            if i == no_of_cols - 1:
                # last triangle, you gotta loop it
                top_left = your_mesh.vectors[net[row_no - 2, i]]
                top_right = your_mesh.vectors[net[row_no - 2, 0]]  ## LOOPED!
                bottom_left = your_mesh.vectors[net[row_no - 1, i]]
            else:
                top_left = your_mesh.vectors[net[row_no - 2, i]]
                top_right = your_mesh.vectors[net[row_no - 2, i + 1]]
                bottom_left = your_mesh.vectors[net[row_no - 1, i]]
            chosen_point = np.array(list(
                set([tuple(x) for x in bottom_left]) & set([tuple(x) for x in top_right]) & set(
                    [tuple(x) for x in top_left])))
            neighbours = [x for (x, y) in enumerate(your_mesh.vectors) if np.all(np.isin(chosen_point, y))]
            if verbose:
                print("working off: {}".format(chosen_point))
                print('Between: {},\n {}\n and {}\n'.format(top_left, top_right, bottom_left))
                print("Found neighbours of point are: {} ".format(neighbours))
            next_triangle_index = [x for x in neighbours if x not in net][0]
            if verbose:
                print("Assigned triangle {}".format(next_triangle_index))
            net[row_no, i] = next_triangle_index
            triangle_indices_not_yet_assigned = set(all_triangle_indices) ^ set(net.flatten())
        if verbose:
            print("{} triangles assigned".format(net[row_no, :]))
            print("triangle indices left to assign are:\n{}".format(triangle_indices_not_yet_assigned))
        return net, triangle_indices_not_yet_assigned

    # row_no = 1
    # col_no=0
    def determining_row_of_downwards_triangles(self,
                                               net, row_no, no_of_cols, your_mesh, verbose=True):
        # this gets all the triangles below the ones in the top row
        top_row = net[row_no, :]
        if verbose:
            print("net is currently:\n{}".format(net))
            print("Working off row {}:\nconsisting of{}".format(row_no, top_row))
        for i in range(len(top_row)):
            next_triangle_index = int(net[row_no, i])  # this is hte triangle above our triangle
            if verbose:
                print("next triangle index is: {}".format(next_triangle_index))
            chosen_triangle = your_mesh.vectors[next_triangle_index]
            neighbours = [y for (y, x) in enumerate(your_mesh.vectors)
                          if np.count_nonzero(np.all(np.isin(x, chosen_triangle), axis=1)) == 2]
            if verbose:
                print("neighbours are: {}".format(neighbours))
            next_triangle_index = [x for x in neighbours if x not in net][0]
            if verbose:
                print("Assigned triangle {}".format(next_triangle_index))
            next_triangle = your_mesh.vectors[next_triangle_index]
            net[row_no + 1, i] = next_triangle_index
        if verbose:
            print("{} triangles assigned".format(net[row_no + 1, :]))
        return net, triangle_indices_not_yet_assigned

    def generate_unfolded_net(self,
                             chosen_triangle_index=0,
                             chosen_direction=0,
                             your_mesh=your_mesh,
                             no_of_rows=16,
                             no_of_cols=5,
                             verbose=False):

        """
        :param chosen_triangle_index: which triangle ot start from
        :param chosen_direction: which direction to unfold in (there are 3)
        :param your_mesh: which icosphere to use
        :param no_of_rows:
        :param no_of_cols:
        :param verbose:
        :return:
        """

        net = first_row_of_net(chosen_triangle_index=chosen_triangle_index,
                               chosen_direction=chosen_direction,
                               no_of_rows=no_of_rows,
                               no_of_cols=no_of_cols,
                               your_mesh=your_mesh,
                               verbose=verbose)
        if verbose:
            print(net)
        # # #### FIRST ROW DONE HOORAY!
        for row_no in range(1, no_of_rows):
            if verbose:
                print("####### Doing row {} now #######".format(row_no))
            if row_no % 2 == 1:
                ## THIS GENERATES THE DOWNWARDS TRIANGLES FROM THE UPWARDS TRIANGLES ABOVE
                determining_row_of_downwards_triangles(net,
                                                       row_no=row_no - 1,
                                                       no_of_cols=no_of_cols,
                                                       your_mesh=your_mesh,
                                                       verbose=verbose)
            else:
                #### THIS IS THE CODE FOR ALL UPRIGHT TRIANGLES AFTER THE FIRST ROW
                determining_row_of_upright_triangles(net,
                                                     row_no=row_no,
                                                     no_of_cols=no_of_cols,
                                                     your_mesh=your_mesh,
                                                     verbose=verbose)

        return net

    def build_colour_list_for_net(self,
                                  net,
                                  no_of_rows,
                                  no_of_cols,
                                  colour_list):
        colour_net = []
        for row in range(no_of_rows):
            # colour_row=[]
            for col in range(no_of_cols):
                if net[row, col] in triangles:
                    idx = triangles.index(net[row, col])
                    colour_net.append(colour_list[idx])
                else:
                    colour_net.append('white')
        return colour_net