import sys
import torch
from typing import NewType
import math
import numpy as np
import torch.nn as nn
import pickle
import trimesh 
import torch.nn.functional as F


Tensor = NewType('Tensor', torch.Tensor)

def solid_angles(
    points: Tensor,
    triangles: Tensor,
    thresh: float = 1e-8
) -> Tensor:

    ''' Compute solid angle between the input points and triangles
        Follows the method described in:
        The Solid Angle of a Plane Triangle
        A. VAN OOSTEROM AND J. STRACKEE
        IEEE TRANSACTIONS ON BIOMEDICAL ENGINEERING,
        VOL. BME-30, NO. 2, FEBRUARY 1983
        Parameters
        -----------
            points: BxQx3
                Tensor of input query points
            triangles: BxFx3x3
                Target triangles
            thresh: float
                float threshold
        Returns
        -------
            solid_angles: BxQxF
                A tensor containing the solid angle between all query points
                and input triangles
    '''
    # Center the triangles on the query points. Size should be BxQxFx3x3
    centered_tris = triangles[:, None] - points[:, :, None, None]

    # BxQxFx3
    norms = torch.norm(centered_tris, dim=-1)

    # Should be BxQxFx3
    cross_prod = torch.cross(
        centered_tris[:, :, :, 1], centered_tris[:, :, :, 2], dim=-1)
    # Should be BxQxF
    numerator = (centered_tris[:, :, :, 0] * cross_prod).sum(dim=-1)
    del cross_prod

    dot01 = (centered_tris[:, :, :, 0] * centered_tris[:, :, :, 1]).sum(dim=-1)
    dot12 = (centered_tris[:, :, :, 1] * centered_tris[:, :, :, 2]).sum(dim=-1)
    dot02 = (centered_tris[:, :, :, 0] * centered_tris[:, :, :, 2]).sum(dim=-1)
    del centered_tris

    denominator = (
        norms.prod(dim=-1) +
        dot01 * norms[:, :, :, 2] +
        dot02 * norms[:, :, :, 1] +
        dot12 * norms[:, :, :, 0]
    )
    del dot01, dot12, dot02, norms

    # Should be BxQ
    solid_angle = torch.atan2(numerator, denominator)
    del numerator, denominator

    torch.cuda.empty_cache()

    return 2 * solid_angle


def compute_vertex_normals(vertices, faces):
    """
    from :https://github.com/ShichenLiu/SoftRas/blob/master/soft_renderer/functional/vertex_normals.py
    :param vertices: [batch size, number of vertices, 3]
    :param faces: [batch size, number of faces, 3]
    :return: [batch size, number of vertices, 3]
    """
    assert (vertices.ndimension() == 3)
    assert (faces.ndimension() == 3 or faces.ndimension() == 2)
    if faces.ndimension() == 2:
        faces = faces.unsqueeze_(0).repeat([vertices.shape[0],1,1])
    assert (vertices.shape[0] == faces.shape[0])
    assert (vertices.shape[2] == 3)
    assert (faces.shape[2] == 3)

    bs, nv = vertices.shape[:2]
    bs, nf = faces.shape[:2]
    device = vertices.device
    normals = torch.zeros(bs * nv, 3).to(device).float()
    faces = faces + (torch.arange(bs).to(device) * nv)[:, None, None] # expanded faces
    vertices_faces = vertices.reshape((bs * nv, 3))[faces.long()]

    faces = faces.view(-1, 3)
    vertices_faces = vertices_faces.view(-1, 3, 3)

    normals.index_add_(0, faces[:, 1].long(),
                       torch.cross(vertices_faces[:, 2] - vertices_faces[:, 1],
                       vertices_faces[:, 0] - vertices_faces[:, 1]).float())
    normals.index_add_(0, faces[:, 2].long(),
                       torch.cross(vertices_faces[:, 0] - vertices_faces[:, 2],
                       vertices_faces[:, 1] - vertices_faces[:, 2]).float())
    normals.index_add_(0, faces[:, 0].long(),
                       torch.cross(vertices_faces[:, 1] - vertices_faces[:, 0],
                       vertices_faces[:, 2] - vertices_faces[:, 0]).float())

    normals = F.normalize(normals, eps=1e-6, dim=1)
    normals = normals.reshape((bs, nv, 3))
    # pytorch only supports long and byte tensors for indexing
    return normals

def winding_numbers(
    points: Tensor,
    triangles: Tensor,
    thresh: float = 1e-8
) -> Tensor:
    ''' Uses winding_numbers to compute inside/outside
        Robust inside-outside segmentation using generalized winding numbers
        Alec Jacobson,
        Ladislav Kavan,
        Olga Sorkine-Hornung
        Fast Winding Numbers for Soups and Clouds SIGGRAPH 2018
        Gavin Barill
        NEIL G. Dickson
        Ryan Schmidt
        David I.W. Levin
        and Alec Jacobson
        Parameters
        -----------
            points: BxQx3
                Tensor of input query points
            triangles: BxFx3x3
                Target triangles
            thresh: float
                float threshold
        Returns
        -------
            winding_numbers: BxQ
                A tensor containing the Generalized winding numbers
    '''
    # The generalized winding number is the sum of solid angles of the point
    # with respect to all triangles.
    return 1 / (4 * math.pi) * solid_angles(
        points, triangles, thresh=thresh).sum(dim=-1)

def get_intersection_mask(vertices, triangles, segments):
    """
        compute status of vertex: inside, outside, or colliding
    """

    bs, nv, _ = vertices.shape

    exterior = torch.zeros((bs, nv), device=vertices.device, dtype=torch.bool)
    exterior = winding_numbers(vertices, triangles).le(0.99)

    for segm_name in segments.names:
        segm_vids = segments.segmentation[segm_name].segment_vidx
        if (exterior[0, segm_vids] == 0).sum() > 0:
            segm_verts = vertices[:, segm_vids, :]
            segm_ext = segments.segmentation[segm_name] \
                .has_self_isect_points(
                    segm_verts.detach(),
                    triangles
            )
            mask = ~segm_ext[0]
            segm_idxs = torch.masked_select(segm_vids, mask)
            exterior[0, segm_idxs] = True

    return exterior

def get_intersection_mask_batch(vertices, triangles, segments):
    """
        compute status of vertex: inside, outside, or colliding
    """

    bs, nv, _ = vertices.shape

    exterior = torch.zeros((bs, nv), device=vertices.device, dtype=torch.bool)
    exterior = winding_numbers(vertices, triangles).le(0.99)
    exterior = segments.batch_has_self_isec_verts(vertices, exterior) 

    return exterior

class BodySegment(nn.Module):
    def __init__(self,
                 handtype,
                 name,
                 faces,
                 segments_folder):
        super(BodySegment, self).__init__()

        self.name = name
        self.append_idx = faces.max().item() 

        sb_path = f"{segments_folder}/segments_bounds.pkl"
        sxseg = pickle.load(open(sb_path, 'rb'))

        # read mesh and find faces of segment
        segment_path = f"{segments_folder}/{handtype}_segment_{name}.ply"
        bandmesh = trimesh.load(segment_path, process=False)
        segment_vidx = torch.from_numpy(np.where(
            np.array(bandmesh.visual.vertex_colors[:,0]) == 255)[0])
        self.register_buffer('segment_vidx', segment_vidx)

        # read boundary information
        self.bands = [x for x in sxseg[name].keys()]
        self.bands_verts = [x for x in sxseg[name].values()]
        self.num_bounds = len(self.bands_verts)
        for idx, bv in enumerate(self.bands_verts):
            self.register_buffer(f'bands_verts_{idx}', torch.tensor(bv))
        self.bands_faces = self.create_band_faces()

        # read mesh and find
        segment_faces_ids = np.where(np.isin(faces.cpu().numpy(),
            segment_vidx).sum(1) == 3)[0]
        segment_faces = torch.cat((faces[segment_faces_ids,:], self.bands_faces), 0)
        self.register_buffer('segment_faces', segment_faces)

        # create vector to select vertices form faces
        tri_vidx = []
        for ii in range(faces.max().item()+1):
            tri_vidx += [torch.nonzero(faces==ii)[0].tolist()]
        self.register_buffer('tri_vidx', torch.tensor(tri_vidx))

    def create_band_faces(self):
        """
            create the faces that close the segment.
        """
        bands_faces = []
        for idx, k in enumerate(self.bands):
            new_vert_idx = self.append_idx + 1 + idx
            new_faces = [[self.bands_verts[idx][i+1], \
                self.bands_verts[idx][i], new_vert_idx] \
                for i in range(len(self.bands_verts[idx])-1)]
            bands_faces += new_faces

        bands_faces_tensor = torch.tensor(
            np.array(bands_faces).astype(np.int64), dtype=torch.long)

        return bands_faces_tensor

    def get_closed_segment(self, vertices):
        """
            create the closed segment mesh from SMPL-X vertices.
        """
        vertices = vertices.detach().clone()
        # append vertices to SMPLX, that close the segment and compute faces
        for idx in range(self.num_bounds):
            bv = eval(f'self.bands_verts_{idx}')
            close_segment_vertices = torch.mean(vertices[:, bv,:], 1,
                                    keepdim=True)
            vertices = torch.cat((vertices, close_segment_vertices), 1)
        segm_triangles = vertices[:, self.segment_faces, :]

        return segm_triangles

    def has_self_isect_verts(self, vertices, thres=0.99):
        """
            check if segment (its vertices) are self intersecting.
        """
        segm_triangles = self.get_closed_segment(vertices)
        segm_verts = vertices[:,self.segment_vidx,:]

        # do inside outside segmentation
        exterior = winding_numbers(segm_verts, segm_triangles).le(0.9)
        # exterior = winding_numbers(segm_verts, segm_triangles).le(thres)

        return exterior

    def has_self_isect_points(self, points, triangles, thres=0.99):
        """
            check if points on segment are self intersecting.
        """
        smplx_verts = triangles[:,self.tri_vidx[:,0], self.tri_vidx[:,1],:]
        segm_triangles = self.get_closed_segment(smplx_verts)

        # do inside outside segmentation
        exterior = winding_numbers(points, segm_triangles).le(thres)

        return exterior

class BatchBodySegment(nn.Module):
    def __init__(self,
                 device,
                 handtype,
                 names,
                 faces,
                 segments_folder):
        super(BatchBodySegment, self).__init__()
        self.names = names
        self.num_segments = len(names)
        self.nv = faces.max().item()

        sb_path = f"{segments_folder}/segments_bounds.pkl"
        sxseg = pickle.load(open(sb_path, 'rb'))

        self.append_idx = [len(b) for a,b in sxseg.items() \
            for c,d in b.items() if a in self.names]
        self.append_idx = np.cumsum(np.array([self.nv] + self.append_idx))

        self.segmentation = {}
        for idx, name in enumerate(names):
            self.segmentation[name] = BodySegment(handtype, name, faces, segments_folder).to(device)

    def batch_has_self_isec_verts(self, vertices, exterior):
        """
            check is mesh is intersecting with itself
        """
        for k, segm in self.segmentation.items():
            if k=='palm':
                exterior[:, segm.segment_vidx] = True
            else:
                exterior_seg = segm.has_self_isect_verts(vertices)
                segm_idxs = torch.masked_select(segm.segment_vidx, ~exterior_seg)
                exterior[:, segm_idxs] = True
        
        return exterior

class contact(object):
    def __init__(self, device, geo_thres=0.02, handtype="RIGHT"):
        ''' 
        Detect inside vertices and compute non-penetration loss

        param geo_thres: a scalar, vertices with geodesicdists smaller than geo_thres are considered as neighbours
        param handtype: string, "RIGHT" or "LEFT"
        param device: torch.device("cuda")
        '''

        # vertex id to divide the whole hand mesh into different hand parts
        self.faces = torch.tensor(np.load("closed_faces.npy").astype(np.int64), dtype=torch.long)
        self.sxseg = pickle.load(open("./segments/segments_bounds.pkl", 'rb'))
        self.segments = BatchBodySegment(
                device, handtype, [x for x in self.sxseg.keys()], self.faces, "./segments",
            )

        # boundary vertex id to close a hand part
        self.bound_ids = torch.tensor(np.load("./bound_ids.npy").astype(np.int64), dtype=torch.long)

        # vertex geodesic distance
        geodesicdists = torch.Tensor(np.load('./geodesicdists.npy'))
        self.geomask = geodesicdists >= geo_thres

        self.device = device

    def computeloss(self, vertices, euc_thres=0.006):
        ''' 
        Detect inside vertices and compute non-penetration loss

        param vertices: Nx778x3, MANO hand vertex positions
        param euc_thres: scalar, imposing nonpenetration loss on those vertices with distance the surface greater than euc_thres
        return nonpenetration_loss: N, penetration loss 
        '''

        # close individual hand part
        vertex_boundary = torch.mean(vertices[:,self.bound_ids], dim=1, keepdim=True)
        vertices_whole = torch.cat([vertices, vertex_boundary], dim=1).clone()
        triangles = vertices_whole[:,self.faces,:]

        bs, nv, _ = vertices_whole.shape

        v2v = vertices_whole.clone().unsqueeze(2).expand(bs, nv, nv, 3) - \
                vertices_whole.clone().unsqueeze(1).expand(bs, nv, nv, 3)
        v2v = torch.norm(v2v, dim=3)

        # detect inside vertices
        with torch.no_grad():
            exterior = get_intersection_mask_batch(
                    vertices=vertices_whole.detach(),
                    triangles=triangles.detach(), 
                    segments=self.segments
                    )

        v2v_mask = v2v.clone()
        v2v_mask[:, ~self.geomask] = 10
        v2v_min, v2v_min_idx = torch.min(v2v_mask, dim=2)
        v2v_min[exterior] = 0

        with torch.no_grad():
            faces = self.faces.to(vertices.device)
            vertex_normals = compute_vertex_normals(vertices_whole, faces)
            vertex_normals_intrude = vertex_normals
            vertex_normals_receive = torch.stack([vertex_normals[b, idx] for b,idx in enumerate(v2v_min_idx)], dim=0)
            vertex_normals_dot = (vertex_normals_intrude*vertex_normals_receive).sum(2)

        # compute non-penetration loss
        v2v_min[vertex_normals_dot>-0.1] = 0
        nonpenetration_loss = torch.clamp(v2v_min-euc_thres, min=0)

        return nonpenetration_loss[:, :-1]>0, nonpenetration_loss.sum(dim=1)

# %%
device = torch.device("cpu") # torch.device("cuda") 
contact_phys = contact(device)
vertices = trimesh.load_mesh("./segments/RIGHT_segment_index.ply").vertices 
vertices = torch.from_numpy(vertices).unsqueeze(0)
inside_vertex, penetration_loss = contact_phys.computeloss(vertices)




