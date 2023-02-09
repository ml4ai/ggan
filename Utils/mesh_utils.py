"""
parse the mesh obj file and return the required matrices:
feature matrix: [n * 6]: n = number of vertices, 6: number of features [x, y, z, normal(x, y, z)]
adjacency matrix: return in sparse cooordinate format
"""
import numpy as np
import torch
import scipy.sparse as sp
from os.path import exists
from pytorch3d.io import load_obj
from itertools import chain
from collections import defaultdict


# Ref: https://github.com/tkipf/pygcn/blob/master/pygcn/utils.py
def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)


def mesh_features(save_path):
    """
    check if the mesh features are already stored in the corresponding location
    if we have the already saved matrices, load them and return
    if not create them and save them in the corresponding location so that they can be used later
    """
    obj_path = save_path + "/" + "model_normalized.obj"
    feature_path = save_path + "/" + "features.npy"
    adj_path = save_path + "/" + "adj.npz"

    if exists(feature_path) and exists(adj_path):
        features = np.load(feature_path)
        adj = sp.load_npz(adj_path)
        features = torch.from_numpy(features)
        adj = sparse_mx_to_torch_sparse_tensor(adj)
        return features, adj

    # else loop over the obj file, create the feature and adj matrices, save them and
    # return them
    print(f"feature and adj tensors not found: creating them...")
    vertices = []
    with open(obj_path, 'r') as obj_file:
        for line in obj_file:
            if line.startswith("v "):
                _, x, y, z = line.strip().split()
                vertices.append([float(x), float(y), float(z)])

    features = np.vstack(vertices).astype(np.float32)
    # index on obj file starts from 1: we need to start our index from 0
    # loop over face lines to find the edges
    rows = []
    cols = []
    with open(obj_path, 'r') as obj_file:
        for line in obj_file:
            if line.startswith("f "):
                _, first_part, second_part, third_part = line.strip().split()
                v1, _, _ = first_part.split("/")
                v2, _, _ = second_part.split("/")
                v3, _, _ = third_part.split("/")

                v1 = int(v1) - 1
                v2 = int(v2) - 1
                v3 = int(v3) - 1

                rows.append(v1)
                cols.append(v2)

                rows.append(v2)
                cols.append(v3)

    rows, cols = zip(*set(zip(rows, cols)))
    data = np.ones(len(rows))
    adj = sp.coo_matrix((data, (np.array(rows), np.array(cols))), shape=(features.shape[0], features.shape[0]),
                        dtype=np.float32)
    # make it symmetric
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    # add self loops
    adj = adj + sp.eye(adj.shape[0], dtype=np.float32)
    # save the features and adj matrix, so we don't need to do it again later
    np.save(feature_path, features)
    sp.save_npz(adj_path, adj)

    adj = sparse_mx_to_torch_sparse_tensor(adj)
    features = torch.from_numpy(features)
    return features, adj


def mesh_features_dual_raw(obj_path):
    """
    check if the mesh features are already stored in the corresponding location
    if we have the already saved matrices, load them and return
    if not create them and save them in the corresponding location so that they can be used later
    """
    folder_path = obj_path.rsplit("/", 1)[0]
    feature_path = folder_path + "/" + "features_dual.npy"
    adj_path = folder_path + "/" + "adj_dual.npz"

    if exists(feature_path) and exists(adj_path):
        features = np.load(feature_path)
        adj = sp.load_npz(adj_path)
        features = torch.from_numpy(features)
        adj = sparse_mx_to_torch_sparse_tensor(adj)
        return features, adj

    # else loop over the obj file, create the feature and adj matrices, save them and
    # return them
    print(f"feature and adj tensors not found: creating them...")
    vertices = []
    normals = []
    with open(obj_path, 'r') as obj_file:
        for line in obj_file:
            if line.startswith("v "):
                _, x, y, z = line.strip().split()
                vertices.append([float(x), float(y), float(z)])

            if line.startswith("vn "):
                _, x, y, z = line.strip().split()
                normals.append([float(x), float(y), float(z)])

    features = []
    temp_dict = defaultdict(list)
    f_count = 0
    with open(obj_path, 'r') as obj_file:
        for line in obj_file:
            if line.startswith("f "):
                _, first_part, second_part, third_part = line.strip().split()
                v1, _, vn1 = first_part.split("/")
                v2, _, vn2 = second_part.split("/")
                v3, _, vn3 = third_part.split("/")

                v1 = int(v1) - 1
                v2 = int(v2) - 1
                v3 = int(v3) - 1

                vn1 = int(vn1) - 1
                vn2 = int(vn2) - 1
                vn3 = int(vn3) - 1
                features.extend(chain(vertices[v1], vertices[v2], vertices[v3],
                                      normals[vn1], normals[vn2], normals[vn3]))

                first_key = str(v1) + "-" + str(v2) if v1 < v2 else str(v2) + "-" + str(v1)
                second_key = str(v2) + "-" + str(v3) if v2 < v3 else str(v3) + "-" + str(v2)
                third_key = str(v1) + "-" + str(v3) if v1 < v3 else str(v3) + "-" + str(v1)
                temp_dict[first_key].append(f_count)
                temp_dict[second_key].append(f_count)
                temp_dict[third_key].append(f_count)
                f_count += 1

    rows = []
    cols = []
    # temp dict that has edge as the keys and the list of faces that use the given edge
    for _, value in temp_dict.items():
        rows.extend(value[:-1])
        cols.extend(value[1:])

    rows, cols = zip(*set(zip(rows, cols)))
    data = np.ones(len(rows))
    adj = sp.coo_matrix((data, (np.array(rows), np.array(cols))), shape=(features.shape[0], features.shape[0]),
                        dtype=np.float32)
    # make it symmetric
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    # add self loops
    adj = adj + sp.eye(adj.shape[0], dtype=np.float32)
    # save the features and adj matrix, so we don't need to do it again later
    np.save(feature_path, features)
    sp.save_npz(adj_path, adj)

    adj = sparse_mx_to_torch_sparse_tensor(adj)
    features = torch.from_numpy(features)
    return features, adj


def mesh_features_dual(save_path, mesh):
    """
    check if the mesh features are already stored in the corresponding location
    if we have the already saved matrices, load them and return
    if not create them and save them in the corresponding location so that they can be used later
    """
    feature_path = save_path + "/" + "features_dual.npy"
    adj_path = save_path + "/" + "adj_dual.npz"

    if exists(feature_path) and exists(adj_path):
        features = np.load(feature_path)
        # features = features[:, :3]
        adj = sp.load_npz(adj_path)
        features = torch.from_numpy(features)
        adj = sparse_mx_to_torch_sparse_tensor(adj)
        return features, adj

    # else loop over the obj file, create the feature and adj matrices, save them and
    # return them
    print(f"feature and adj tensors not found: creating them...")
    # verts, faces, aux = load_obj(obj_path)
    verts = mesh.verts_packed()
    faces = mesh.faces_packed()
    # verts_normals = mesh.verts_normals_packed()
    faces_normals = mesh.faces_normals_packed()
    features = []
    for idx, item in enumerate(faces):
        v1, v2, v3 = [int(x) for x in item]

        v1_x, v1_y, v1_z = [x.item() for x in verts[v1]]
        v2_x, v2_y, v2_z = [x.item() for x in verts[v2]]
        v3_x, v3_y, v3_z = [x.item() for x in verts[v3]]

        v_x = (v1_x + v2_x + v3_x) / 3.0
        v_y = (v1_y + v2_y + v3_y) / 3.0
        v_z = (v1_z + v2_z + v3_z) / 3.0

        n_x, n_y, n_z = [x.item() for x in faces_normals[idx]]
        features.append([v_x, v_y, v_z, n_x, n_y, n_z])

    features = np.vstack(features).astype(np.float32)
    rows = []
    cols = []
    # temp dict that has edge as the keys and the list of faces that use the given edge
    temp_dict = defaultdict(list)
    for idx, item in enumerate(faces):
        v1, v2, v3 = [str(x.item()) for x in item]
        first_key = v1 + "-" + v2 if v1 < v2 else v2 + "-" + v1
        second_key = v2 + "-" + v3 if v2 < v3 else v3 + "-" + v2
        third_key = v1 + "-" + v3 if v1 < v3 else v3 + "-" + v1
        temp_dict[first_key].append(idx)
        temp_dict[second_key].append(idx)
        temp_dict[third_key].append(idx)

    for _, value in temp_dict.items():
        rows.extend(value[:-1])
        cols.extend(value[1:])

    rows, cols = zip(*set(zip(rows, cols)))
    data = np.ones(len(rows))
    adj = sp.coo_matrix((data, (np.array(rows), np.array(cols))), shape=(features.shape[0], features.shape[0]),
                        dtype=np.float32)
    # make it symmetric
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    # add self loops
    adj = adj + sp.eye(adj.shape[0], dtype=np.float32)
    # save the features and adj matrix, so we don't need to do it again later
    np.save(feature_path, features)
    sp.save_npz(adj_path, adj)

    adj = sparse_mx_to_torch_sparse_tensor(adj)
    features = torch.from_numpy(features)
    return features, adj
