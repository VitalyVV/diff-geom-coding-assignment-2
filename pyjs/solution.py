import numpy as np
import math
import drawwh
import scipy.sparse


class Mesh:
    def __init__(self, faces, coordinates = None):
        self.faces = faces
        vertices = set(i for f in faces for i in f)
        self.n = max(vertices)+1
        if coordinates != None:
            self.coordinates = np.array(coordinates)

        assert set(range(self.n)) == vertices
        for f in faces:
            assert len(f)==3
        if coordinates != None:
            assert self.n == len(coordinates)
            for c in coordinates:
                assert len(c)==3
    
    @classmethod
    def fromobj(cls, filename):
        faces, vertices = drawwh.obj_read(filename)
        return cls(faces, vertices)

    def draw(self):
        drawwh.draw(self.faces, self.coordinates.tolist())
        

    def angleDefect(self, vertex): # vertex is an integer (vertex index from 0 to self.n-1)
        # TODO: Not to do
        raise NotImplementedError

    def build_link(self, v):
        def get_edges(face):
            return (face[0], face[1]), (face[1], face[2]), (face[2], face[0])

        def get_opposite_edge(edges, v):
            for edge in edges:
                if v not in edge:
                    return edge

        contain = [f for f in self.faces if v in f]
        edges = []
        # Take edge without `v'
        for face in contain:
            edgs = get_edges(face)
            edges.append(get_opposite_edge(edgs, v))

        # for all that contain v
        # Enumerate it
        link = list(enumerate(edges))
        # return link
        return link


    def get_angles(self, i, j, link):
        n = [x for x in link if x[1]==j]
        init = np.hypot(self.coordinates[i-1], self.coordinates[j-1])
        prev = link[n[0][0]-1][1]-1 if n[0][0]-1 >= 0 else link[len(link)-1][1]-1
        next = link[n[0][0]+1][1] if n[0][0]+1 < len(link) else link[len(link)-1][1]
        alpha = np.hypot(self.coordinates[prev], self.coordinates[i-1]) / init
        betha = self.coordinates[next]




    def LaplaceOperator(self, anchors = None, anchor_weight = 1.): # anchors is a list of vertex indices, anchor_weight is a positive number
        if anchors is None:
            anchors = []

        vertices = set(i for f in self.faces for i in f)
        matr = scipy.sparse.csr_matrix((self.n, self.n))

        for i in range(self.n):
            for vertex in self.n_i(i):
                link_v = self.build_link(i)

                #alpha, betha = get angles()
                matr[i][vertex] = 0.5 * (
                        np.cos(matr[i][vertex]) / np.sin(matr[i][vertex]) +
                        np.cos(matr[i][vertex]) / np.sin(matr[i][vertex]))

        for i in range(self.n):
            sumo = 0
            for vertex in self.n_i(i):
                sumo += matr[i][vertex]
            matr[i][i] = sumo


        raise NotImplementedError

    def get_all_edges(self):
        edges = []
        for f in self.faces:
            edges.append((f[0], f[1]))
            edges.append((f[1], f[2]))
            edges.append((f[2], f[0]))

        return edges

    def n_i(self, vert):
        def get_opposite_vertex(edge, vertex):
            return edge[0] if vertex == edge[1] else edge[1]
        edges = self.get_all_edges()
        nofi = []
        for edge in edges:
           if vert in edge:
               nofi.append(get_opposite_vertex(edge, vert))

        return nofi

    def smoothen(self):
        raise NotImplementedError

    def transform(self, anchors, anchor_coordinates, anchor_weight = 1.):# anchors is a list of vertex indices, anchor_coordinates is a list of same length of vertex coordinates (arrays of length 3), anchor_weight is a positive number
        raise NotImplementedError

def dragon(): #
    mesh = Mesh.fromobj("dragon.obj")
    mesh.draw()
