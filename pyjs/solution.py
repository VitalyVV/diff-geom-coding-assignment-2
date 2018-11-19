import drawwh
import numpy as np
import scipy.sparse
import scipy.sparse.linalg


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
        """
        Calculate angles between neighbors between (ith, jth) edge.
            j*
          /  | \
        *a _*i _*b    ->  a, b is the needed angles.
         \      /
          \   /
            *
        :param i: ith vertex
        :param j: jth vertex
        :param link: link including jth vertex over ith vertex
        :return: cotan(alpha), cotan(beta)
        """
        def hypot(x, y):
            """
            np.hypot do it elementwise and not returning required result
            """
            return np.sqrt((x[0] - y[0]) ** 2 + (x[1] - y[1]) ** 2 + (x[2] - y[2]) ** 2)

        # TO BE REFACTORED
        # Use list of one item to proceed, this item can be obtained many different ways
        n = [x for x in link if j == x[1][0]]

        init  = hypot(self.coordinates[i], self.coordinates[j])
        prev  = link[n[0][0]-1][0] if n[0][0]-1 >= 0 else link[len(link)-1][0]
        next_ = link[n[0][0]+1][0] if n[0][0]+1 < len(link) else link[0][0]
        alpha = hypot(self.coordinates[prev], self.coordinates[i]) / init
        betha = hypot(self.coordinates[next_], self.coordinates[i]) / init
        return alpha, betha


    def cotan(self, i, j):
        link_v = self.build_link(i)
        alpha, betha = self.get_angles(i, j, link_v)

        return 0.5 * (alpha + betha)


    def LaplaceOperator(self, anchors = None, anchor_weight = 1.): # anchors is a list of vertex indices, anchor_weight is a positive number
        if anchors is None:
            anchors = []

        vertices = set(i for f in self.faces for i in f)
        matr = scipy.sparse.csr_matrix((self.n, self.n))

        for i in vertices:
            for j in self.n_i(i):
                matr[i,j] = self.cotan(i, j)

        # TO REFACTOR
        # It is very ad hoc to calculate sum like this
        for i in range(self.n):
            matr[i, i] = sum(matr[i].data)


        self.laplace_operator = matr
        self.compute_inversion()


    def get_all_edges(self):
        edges = []
        for f in self.faces:
            edges.append((f[0], f[1]))
            edges.append((f[1], f[2]))
            edges.append((f[2], f[0]))

        return edges

    def n_i(self, vert):
        """
        :param vert:
        :return: list of vertices indexes
        """

        # TO REFACTOR:
        # It can be written as an local function def ...
        get_opposite_vertex = lambda e, v: e[0] if v == e[1] else e[1]


        edges = self.get_all_edges()
        nofi = []
        for edge in edges:
           if vert in edge:
               nofi.append(get_opposite_vertex(edge, vert))

        return nofi


    def compute_inversion(self):
        self.inversion = scipy.sparse.linalg.inv(self.laplace_operator.transpose() * self.laplace_operator) * self.laplace_operator


    def forwardEuler(self, h):

        # TO MAYBE REFACTOR
        # Probably can be done without splitting to axises and use different approach overall
        eps = 0.0001
        f = self.coordinates
        f[:, 0] = self.inversion * self.coordinates[:, 0]
        f[:, 1] = self.inversion * self.coordinates[:, 1]
        f[:, 2] = self.inversion * self.coordinates[:, 2]
        while (self.coordinates - f).any() > eps:
            self.coordinates[:, 0] += h * self.laplace_operator * self.coordinates[:, 0]
            self.coordinates[:, 1] += h * self.laplace_operator * self.coordinates[:, 1]
            self.coordinates[:, 2] += h * self.laplace_operator * self.coordinates[:, 2]
            self.draw()

    def backwardEuler(self, h):
        eps = 0.0001
        f = self.coordinates
        f[:, 0] = self.inversion * self.coordinates[:, 0]
        f[:, 1] = self.inversion * self.coordinates[:, 1]
        f[:, 2] = self.inversion * self.coordinates[:, 2]
        I = np.identity(self.n) # TO MAYBE REFACTOR: Can be self.laplace_operator.shape[0] instead self.n
        step = I - h * self.laplace_operator
        while (self.coordinates - f).any() > eps:
            self.coordinates[:, 0] = step * self.coordinates[:, 0]
            self.coordinates[:, 1] = step * self.coordinates[:, 1]
            self.coordinates[:, 2] = step * self.coordinates[:, 2]
            self.draw()


    def smoothen(self):
        if self.laplace_operator is None:
            raise NotImplementedError
        else:
            self.forwardEuler(0.001)


    def transform(self, anchors, anchor_coordinates, anchor_weight = 1.):# anchors is a list of vertex indices, anchor_coordinates is a list of same length of vertex coordinates (arrays of length 3), anchor_weight is a positive number
        raise NotImplementedError

def dragon(): #
    mesh = Mesh.fromobj("teddy.obj")
    mesh.LaplaceOperator()
    mesh.smoothen()
    mesh.draw()


if __name__ == '__main__':
    dragon()