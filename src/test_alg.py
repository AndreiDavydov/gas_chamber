import numpy as np
import cv2
import matplotlib.pyplot as plt

from src.alg import get_edges, make_decision_on_one_component

def _test_comp_update(idx, state):
    """
    state : State object
    idx : index of the state to test
    """

    states = state.get_states()
    diffs = state.get_diffs()

    x = state.interms[idx].astype(np.uint8)
    edgeO, edgeI = get_edges(x)

    y = diffs[idx].astype(np.uint8)
    num_labels, labels = cv2.connectedComponents(y)
    comps = [(labels == i).astype(np.uint8) for i in range(1, num_labels)]

    im = np.zeros((y.shape[0], y.shape[1], 3), dtype=np.float32)

    # comps = comps[:1]
    if len(comps) > 0:
        fig, ax = plt.subplots(len(comps), 1, figsize=(im.shape[1] / 100 * 2 / 1.5, im.shape[0] / 100 * len(comps)*2, ), dpi=100)
        if len(comps) == 1: ax = [ax]

        for i, comp in enumerate(comps):
            im[..., 0] = edgeI.astype(np.float32)
            im[..., 1] = comp.astype(np.float32)
            im[..., 2] = edgeO.astype(np.float32)
            
            ax[i].imshow(im)
            ax[i].axis('off')

            decision = make_decision_on_one_component(x, comp, edgeO, edgeI)

            ax[i].set_title(f"{decision.message}, {decision.note}")

    return fig