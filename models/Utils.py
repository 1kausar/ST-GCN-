### Reference from: https://github.com/yysijie/st-gcn/blob/master/net/utils/graph.py

import numpy as np


class Graph:
    """The Graph to model the skeleton extracted by MediaPipe (Pose + Hands).
    Args:
        - strategy: (string) must be one of the follow candidates
            - uniform: Uniform Labeling,
            - distance: Distance Partitioning,
            - spatial: Spatial Configuration,
        For more information, please refer to the section 'Partition Strategies'
            in the ST-GCN paper (https://arxiv.org/abs/1801.07455).
        - layout: (string) only 'body_hand' shomorthito - 75 node:
            33 MediaPipe Pose + 21 left-hand + 21 right-hand.
        - max_hop: (int) the maximal distance between two connected nodes.
        - dilation: (int) controls the spacing between the kernel points.
    """
    def __init__(self,
                 layout='body_hand',
                 strategy='spatial',
                 max_hop=3,
                 dilation=1):
        self.max_hop = max_hop
        self.dilation = dilation

        self.get_edge(layout)
        self.hop_dis = get_hop_distance(self.num_node, self.edge, max_hop)
        # Center (nose) theke prottek node-er ashol distance - max_hop-er upor-e ceiling nai.
        # (hop_dis shudhu max_hop porjonto; er baire inf hoye 'spatial' partition bhenge dito)
        self.center_dis = get_hop_distance(self.num_node, self.edge,
                                           self.num_node)[:, self.center]
        self.get_adjacency(strategy)

    def get_edge(self, layout):
        if layout == 'body_hand':
            # 33 MediaPipe Pose landmark + 21 left-hand + 21 right-hand = 75 node.
            # Node 0-32   : MediaPipe Pose (same order/index as MediaPipe)
            # Node 33-53  : left hand  (MediaPipe Hands, 21 point, wrist = 33)
            # Node 54-74  : right hand (MediaPipe Hands, 21 point, wrist = 54)
            self.num_node = 75
            self_link = [(i, i) for i in range(self.num_node)]

            pose_edge = [(0, 1), (0, 4), (1, 2), (2, 3), (3, 7), (4, 5), (5, 6), (6, 8),
                        (9, 10), (11, 12), (11, 13), (11, 23), (12, 14), (12, 24),
                        (13, 15), (14, 16), (15, 17), (15, 19), (15, 21), (16, 18),
                        (16, 20), (16, 22), (17, 19), (18, 20), (23, 24), (23, 25),
                        (24, 26), (25, 27), (26, 28), (27, 29), (27, 31), (28, 30),
                        (28, 32), (29, 31), (30, 32)]
            hand_edge = [(0, 1), (0, 17), (1, 2), (1, 5), (2, 3), (3, 4), (5, 6), (5, 9),
                        (6, 7), (7, 8), (9, 10), (9, 13), (10, 11), (11, 12), (13, 14),
                        (13, 17), (14, 15), (15, 16), (17, 18), (18, 19), (19, 20)]
            left_hand_edge = [(a + 33, b + 33) for a, b in hand_edge]
            right_hand_edge = [(a + 54, b + 54) for a, b in hand_edge]
            # pose wrist (15=left, 16=right) -> hand wrist (33=left, 54=right)
            bridge = [(15, 33), (16, 54)]

            # Face <-> body connection. MediaPipe-er default edge-e mukh (0-10) ar
            # shoulder (11,12) alada thake, tai nose (center) theke body/hand pouchay na
            # ar 'spatial' partition (root/close/further) kaj kore na. Tai ei 4-ta edge.
            face_body = [(0, 9), (0, 10), (0, 11), (0, 12)]

            neighbor_link = pose_edge + face_body + left_hand_edge + right_hand_edge + bridge
            self.edge = self_link + neighbor_link
            self.center = 0   # nose
        else:
            raise ValueError("This layout is not supported! Only 'body_hand' (MediaPipe, 75 node) ache.")

    def get_adjacency(self, strategy):
        valid_hop = range(0, self.max_hop + 1, self.dilation)
        adjacency = np.zeros((self.num_node, self.num_node))
        for hop in valid_hop:
            adjacency[self.hop_dis == hop] = 1
        normalize_adjacency = normalize_digraph(adjacency)

        if strategy == 'uniform':
            A = np.zeros((1, self.num_node, self.num_node))
            A[0] = normalize_adjacency
            self.A = A
        elif strategy == 'distance':
            A = np.zeros((len(valid_hop), self.num_node, self.num_node))
            for i, hop in enumerate(valid_hop):
                A[i][self.hop_dis == hop] = normalize_adjacency[self.hop_dis ==
                                                                hop]
            self.A = A
        elif strategy == 'spatial':
            A = []
            for hop in valid_hop:
                a_root = np.zeros((self.num_node, self.num_node))
                a_close = np.zeros((self.num_node, self.num_node))
                a_further = np.zeros((self.num_node, self.num_node))
                for i in range(self.num_node):
                    for j in range(self.num_node):
                        if self.hop_dis[j, i] == hop:
                            if self.center_dis[j] == self.center_dis[i]:
                                a_root[j, i] = normalize_adjacency[j, i]
                            elif self.center_dis[j] > self.center_dis[i]:
                                a_close[j, i] = normalize_adjacency[j, i]
                            else:
                                a_further[j, i] = normalize_adjacency[j, i]
                if hop == 0:
                    A.append(a_root)
                else:
                    A.append(a_root + a_close)
                    A.append(a_further)
            A = np.stack(A)
            self.A = A
            #self.A = np.swapaxes(np.swapaxes(A, 0, 1), 1, 2)
        else:
            raise ValueError("This strategy is not supported!")


def get_hop_distance(num_node, edge, max_hop=1):
    A = np.zeros((num_node, num_node))
    for i, j in edge:
        A[j, i] = 1
        A[i, j] = 1

    # compute hop steps
    hop_dis = np.zeros((num_node, num_node)) + np.inf
    transfer_mat = [np.linalg.matrix_power(A, d) for d in range(max_hop + 1)]
    arrive_mat = (np.stack(transfer_mat) > 0)
    for d in range(max_hop, -1, -1):
        hop_dis[arrive_mat[d]] = d
    return hop_dis


def normalize_digraph(A):
    Dl = np.sum(A, 0)
    num_node = A.shape[0]
    Dn = np.zeros((num_node, num_node))
    for i in range(num_node):
        if Dl[i] > 0:
            Dn[i, i] = Dl[i]**(-1)
    AD = np.dot(A, Dn)
    return AD


def normalize_undigraph(A):
    Dl = np.sum(A, 0)
    num_node = A.shape[0]
    Dn = np.zeros((num_node, num_node))
    for i in range(num_node):
        if Dl[i] > 0:
            Dn[i, i] = Dl[i]**(-0.5)
    DAD = np.dot(np.dot(Dn, A), Dn)
    return DAD
