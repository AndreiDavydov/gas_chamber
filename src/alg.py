import cv2
import numpy as np
from attrdict import AttrDict


def prep_state0(tube_mask, kernel_size=(20,20)):
    """
    Given a tube mask, prepare the state0 as nonzero values along the tube.
    """
    tube_mask = tube_mask.astype(np.float32)
    kernel_outer = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, kernel_size)
    kernel_inner = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
    outer = cv2.dilate(tube_mask, kernel_outer, iterations=1)
    inner = cv2.dilate(tube_mask, kernel_inner, iterations=1)
    state0 = outer - inner
    state0 = state0.astype(np.uint8)
    return state0


def get_edge(x, kernel_size=(5,5)):
    """
    Get the edge of the image using morphological operations
    """
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, kernel_size)
    wider = cv2.dilate(x, kernel, iterations=1)
    eroded = cv2.erode(x, kernel, iterations=1)
    edge = wider - eroded
    return edge


def get_edges(x, **kwargs):
    """
    Get the inner and outer edges of the image using morphological operations
    """
    kernel_size = kwargs.get("kernel_size", (5,5))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, kernel_size)
    wider = cv2.dilate(x, kernel, iterations=1)
    eroded = cv2.erode(x, kernel, iterations=1)
    return wider - x, x - eroded
    

def denoise_and_fill(x, **kwargs):
    """
    Denoises the image and fills small holes in the image
    Denoising includes the following morpholgy operations:
    1. Removing noise via opening with small kernel #(3,3)
    2. Filling small holes via closing with large kernel #(5,5)
    3. Remove larger noise via opening with large kernel #(5,5)
    """
    open_kernel_size = kwargs.get("open_kernel_size", (3,3))
    open_iterations = kwargs.get("open_iterations", 1)
    close_kernel_size = kwargs.get("close_kernel_size", (5,5))
    close_iterations = kwargs.get("close_iterations", 3)

    x = x.copy()
    # denoise: removes noise via opening
    kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, open_kernel_size)
    x = cv2.morphologyEx(x, cv2.MORPH_OPEN, kernel_open, iterations=open_iterations)
    
    # fill small holes via closing
    kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, close_kernel_size)
    x = cv2.morphologyEx(x, cv2.MORPH_CLOSE, kernel_close, iterations=close_iterations)

    x = cv2.morphologyEx(x, cv2.MORPH_OPEN, kernel_close, iterations=open_iterations)

    return x


def make_decision_on_one_component(x, comp, edgeO, edgeI, **kwargs):
    """
    Given a component of the difference image, make a decision on whether to add it to the state, subtract or ignore.

    x : H x W, binary mask of the current state
    comp : component of the difference image
    edgeO : outer edge of the state
    edgeI : inner edge of the state

    Case 1: completely inside: do nothing
    Case 2: completely inside but partially in the edgeI
        Case 2a : if comp[edgeI] <= T_I, ignore
        Case 2b : if comp[edgeI] > T_I, subtract
    Case 3: completely inside outer edge
        Case 3a : if comp[edgeO] <= T_O, subtract
        Case 3b : if comp[edgeO] > T_O, add
    """
    T_I = kwargs.get("T_I", 0.05)
    T_O = kwargs.get("T_O", 0.3)

    edgeI_inters = np.logical_and(edgeI, comp).sum()
    edgeO_inters = np.logical_and(edgeO, comp).sum()
    edge_inters = edgeI_inters + edgeO_inters
    x_inters = np.logical_and(x, comp).sum()
    comp_sum = comp.sum()
    
    if x_inters == comp_sum and edge_inters == 0:
        # case 1: completely inside: do nothing
        return AttrDict(message="ignore", note=f"case 1: inters with x={x_inters}, edgeI={edgeI_inters}, edgeO={edgeO_inters}, comp_sum={comp_sum}")

    elif x_inters == comp_sum and edge_inters > 0:
        # case 2: inside but near the edge
        if edge_inters / comp_sum <= T_I:
            # case 2a: ignore, intersection is small
            return AttrDict(message="ignore", note=f"case 2a: inters with x={x_inters}, edgeI={edgeI_inters}, edgeO={edgeO_inters}, comp_sum={comp_sum}")
        else:
            # case 2b: shrink, significant chunk of the component is near the edge
            return AttrDict(message="subtract", note=f"case 2b: inters with x={x_inters}, edgeI={edgeI_inters}, edgeO={edgeO_inters}, comp_sum={comp_sum}")

    elif edgeO_inters > 0 and x_inters + edgeO_inters == comp_sum:
        # case 3: completely inside outer edge
        if edgeO_inters / comp_sum <= T_O:
            # case 3a: more in than out, shrink
            return AttrDict(message="subtract", note=f"case 3a: inters with x={x_inters}, edgeI={edgeI_inters}, edgeO={edgeO_inters}, comp_sum={comp_sum}")

        else:
            # case 3b: more out than in, add
            return AttrDict(message="add", note=f"case 3b: inters with x={x_inters}, edgeI={edgeI_inters}, edgeO={edgeO_inters}, comp_sum={comp_sum}")
    
    elif x_inters > 0 and x_inters + edgeO_inters < comp_sum:
        # case 4: has some parts in the state and some parts outside
        outer = (comp_sum - (x_inters + edgeO_inters))
        if outer / comp_sum <= T_O:
            # case 4a: more in than out, shrink
            return AttrDict(message="subtract", note=f"case 4a: inters with x={x_inters}, edgeI={edgeI_inters}, edgeO={edgeO_inters}, comp_sum={comp_sum}")
        else:
            # case 4b: more out than in, add
            return AttrDict(message="add", note=f"case 4b: inters with x={x_inters}, edgeI={edgeI_inters}, edgeO={edgeO_inters}, comp_sum={comp_sum}")

    else:
        # case 5: completely outside, add
        return AttrDict(message="add", note=f"case 5: inters with x={x_inters}, edgeI={edgeI_inters}, edgeO={edgeO_inters}, comp_sum={comp_sum}")


def update_binary_state(x, y, **kwargs):
    """
    x : H x W, binary mask of the current state
    y : H x W, binary mask of the difference
    """
    # find edges of x
    edgeO, edgeI = get_edges(x, **kwargs)

    # split y in connected components
    num_labels, labels = cv2.connectedComponents(y)
    comps = [(labels == i).astype(np.uint8) for i in range(1, num_labels)]

    for comp in comps:
        decision = make_decision_on_one_component(x, comp, edgeO, edgeI, **kwargs)

        if decision.message == "add":
            x = x + comp
            x[x > 1] = 1
        
        elif decision.message == "subtract":
            x = x * (1 - comp)

        elif decision.message == "ignore":
            pass
        
    return x


class State():
    def __init__(self, tube_mask, edge_kernel_size=(10,10), add_tube_edges_every=True, output_kernel_size=(9,9)):

        self.add_tube_edges_every = add_tube_edges_every
        self.edge_kernel_size = edge_kernel_size
        self.output_kernel_size = output_kernel_size

        self.state0 = prep_state0(tube_mask, edge_kernel_size)
        self.state = self.state0.copy()
        self.tube_mask = tube_mask
        
        self.states = [self.state]
        self.diffs = []

        self.interms = []

    def add_tube_edges(self, x):
        """
        Add the tube edges to the state
        """
        x = x + self.state0
        x[x > 1] = 1
        return x

    def _update(self, diff_raw, **kwargs):
        """
        Update the state with the new difference frame

        The state consists of clusters of non-zero pixels. Diff is the difference between the current frame and the previous frame, which also represents the clusters of change.
        """
        denoise_output = kwargs.get("denoise_output", True)
        x = self.state.copy()

        x = denoise_and_fill(x, **kwargs)
        if self.add_tube_edges_every: x = self.add_tube_edges(x)
        self.interms.append(x) # for debug

        y = denoise_and_fill(diff_raw, **kwargs)
        self.diffs.append(y) # for debug
        
        x = update_binary_state(x, y, **kwargs)

        if denoise_output:
            # fill the output with larger kernel
            x = denoise_and_fill(x, close_kernel_size=self.output_kernel_size)

        if self.add_tube_edges_every: x = self.add_tube_edges(x)
        return x

    def update(self, diff_raw, **kwargs):
        """
        Update the state with the new difference frame
        """
        self.state = self._update(diff_raw, **kwargs)
        self.states.append(self.state)
        return self.state
    
    def get_states(self):
        return np.stack(self.states)
    
    def get_diffs(self):
        return np.stack(self.diffs)
    





