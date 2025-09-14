from copy import deepcopy
import numpy as np
import torch
from scipy.optimize import linear_sum_assignment

class PermutationAligner:
    """
    Allinea i neuroni di model_b a model_a usando il metodo Git Re-Basin.
    Questa versione è generale e gestisce architetture sia FC che CNN in modo corretto.
    """
    def __init__(self, model_arch):
        self.model_arch = model_arch
        self.layers_to_align = self.model_arch().layers_to_align

    def _get_final_classifier_name(self, sd):
        all_keys = list(sd.keys())
        last_weight_key = [k for k in all_keys if 'weight' in k][-1]
        return last_weight_key.split('.')[0]

    def align(self, sd_a, sd_b, return_transformations=False):
        print("Inizio allineamento con Permutazioni (Git Re-Basin)...")
        aligned_sd_b = deepcopy(sd_b)
        permutations = {}

        for iteration in range(5): # Iteriamo per far convergere le permutazioni
            changed = False
            for i, layer_name in enumerate(self.layers_to_align):
                # Estraiamo i pesi (W) e li appiattiamo lungo le dimensioni di input
                W_a = sd_a[f"{layer_name}.weight"]
                W_b = aligned_sd_b[f"{layer_name}.weight"]

                # Calcolo della correlazione dei pesi in ingresso
                cost_matrix = -torch.matmul(W_a.reshape(W_a.shape[0], -1), W_b.reshape(W_b.shape[0], -1).T)

                # Trova il layer successivo
                is_last_hidden_layer = (i == len(self.layers_to_align) - 1)
                if not is_last_hidden_layer:
                    next_layer_name = self.layers_to_align[i+1]
                else:
                    next_layer_name = self._get_final_classifier_name(sd_a)

                W_next_a = sd_a[f"{next_layer_name}.weight"]
                W_next_b = aligned_sd_b[f"{next_layer_name}.weight"]

                # Calcolo della correlazione dei pesi in uscita
                if W_next_a.dim() > 2: # Layer successivo è Conv
                    # Permutiamo per isolare il canale di input (dim 1)
                    W_next_a_flat = W_next_a.permute(1, 0, 2, 3).reshape(W_next_a.shape[1], -1)
                    W_next_b_flat = W_next_b.permute(1, 0, 2, 3).reshape(W_next_b.shape[1], -1)
                    cost_matrix -= torch.matmul(W_next_a_flat, W_next_b_flat.T)
                else: # Layer successivo è FC
                    if W_a.dim() > 2: # Transizione Conv -> FC
                        # Appiattiamo il peso del layer FC per farlo corrispondere ai canali di output del Conv
                        num_groups = W_a.shape[0] # Num canali di output del layer conv precedente
                        W_next_a_flat = W_next_a.reshape(W_next_a.shape[0], num_groups, -1).permute(1, 0, 2).reshape(num_groups, -1)
                        W_next_b_flat = W_next_b.reshape(W_next_b.shape[0], num_groups, -1).permute(1, 0, 2).reshape(num_groups, -1)
                        cost_matrix -= torch.matmul(W_next_a_flat, W_next_b_flat.T)
                    else: # Transizione FC -> FC
                        cost_matrix -= torch.matmul(W_next_a.T, W_next_b)

                # Risolviamo il problema di assegnazione
                row_ind, col_ind = linear_sum_assignment(cost_matrix.cpu().numpy())
                permutations[layer_name] = torch.tensor(col_ind).to(W_a.device)

                if not np.array_equal(col_ind, np.arange(len(col_ind))):
                    changed = True

                # Applichiamo la permutazione
                aligned_sd_b[f"{layer_name}.weight"] = W_b[col_ind, :]
                aligned_sd_b[f"{layer_name}.bias"] = aligned_sd_b[f"{layer_name}.bias"][col_ind]

                # Compensiamo nel layer successivo
                W_next_b_comp = aligned_sd_b[f"{next_layer_name}.weight"]
                if W_next_b_comp.dim() > 2: # Il layer successivo è Conv
                    aligned_sd_b[f"{next_layer_name}.weight"] = W_next_b_comp[:, col_ind, :, :]
                else: # Il layer successivo è FC
                    if W_b.dim() > 2: # Transizione Conv -> FC
                        num_groups = W_b.shape[0]
                        W_next_b_reshaped = W_next_b_comp.reshape(W_next_b_comp.shape[0], num_groups, -1)
                        W_next_b_permuted = W_next_b_reshaped[:, col_ind, :]
                        aligned_sd_b[f"{next_layer_name}.weight"] = W_next_b_permuted.reshape(W_next_b_comp.shape)
                    else: # Transizione FC -> FC
                         aligned_sd_b[f"{next_layer_name}.weight"] = W_next_b_comp[:, col_ind]

            if not changed and iteration > 0:
                print(f"Convergenza raggiunta dopo {iteration+1} iterazioni.")
                break

        if return_transformations:
            return aligned_sd_b, permutations
        return aligned_sd_b


class OrthogonalAligner:
    """
    Allinea i neuroni di model_b a model_a usando l'analisi di Procrustes.
    Questa versione è generale e gestisce architetture sia FC che CNN in modo corretto.
    """
    def __init__(self, model_arch):
        self.model_arch = model_arch
        self.layers_to_align = self.model_arch().layers_to_align

    def _get_final_classifier_name(self, sd):
        all_keys = list(sd.keys())
        last_weight_key = [k for k in all_keys if 'weight' in k][-1]
        return last_weight_key.split('.')[0]

    def align(self, sd_a, sd_b, return_transformations=False):
        print("Inizio allineamento con Matrici Ortogonali (Procrustes)...")
        aligned_sd_b = deepcopy(sd_b)
        rotations = {}

        for i, layer_name in enumerate(self.layers_to_align):
            W_a = sd_a[f"{layer_name}.weight"]
            W_b = aligned_sd_b[f"{layer_name}.weight"]

            # Appiattiamo sempre la matrice dei pesi in [out_dim, in_dim_flat]
            W_a_flat = W_a.reshape(W_a.shape[0], -1)
            W_b_flat = W_b.reshape(W_b.shape[0], -1)

            # Calcoliamo la rotazione ottimale Q
            M = torch.matmul(W_a_flat, W_b_flat.T)
            U, _, V_t = torch.linalg.svd(M, full_matrices=False)
            Q = torch.matmul(U, V_t)
            rotations[layer_name] = Q

            # Applichiamo la rotazione ai pesi e al bias del layer corrente
            rotated_W_b = torch.matmul(Q, W_b_flat)
            aligned_sd_b[f"{layer_name}.weight"] = rotated_W_b.reshape(W_b.shape)
            aligned_sd_b[f"{layer_name}.bias"] = torch.matmul(Q, aligned_sd_b[f"{layer_name}.bias"])

            # Compensiamo nel layer successivo
            is_last_hidden_layer = (i == len(self.layers_to_align) - 1)
            if not is_last_hidden_layer:
                next_layer_name = self.layers_to_align[i+1]
            else:
                next_layer_name = self._get_final_classifier_name(sd_a)

            W_next_b = aligned_sd_b[f"{next_layer_name}.weight"]

            if W_next_b.dim() > 2: # Layer successivo è Conv
                W_next_b_reshaped = W_next_b.permute(0, 2, 3, 1) # Mette il canale di input alla fine
                rotated_W_next_b = torch.matmul(W_next_b_reshaped, Q.T)
                aligned_sd_b[f"{next_layer_name}.weight"] = rotated_W_next_b.permute(0, 3, 1, 2) # Riporta alla forma originale
            else: # Layer successivo è FC
                if W_b.dim() > 2: # Transizione Conv -> FC
                    num_groups = W_b.shape[0]
                    W_next_b_reshaped = W_next_b.reshape(W_next_b.shape[0], num_groups, -1)
                    W_next_b_permuted = W_next_b_reshaped.permute(0, 2, 1) # Mette i gruppi di input alla fine
                    rotated_W_next_b = torch.matmul(W_next_b_permuted, Q.T)
                    aligned_sd_b[f"{next_layer_name}.weight"] = rotated_W_next_b.permute(0, 2, 1).reshape(W_next_b.shape)
                else: # Transizione FC -> FC
                    aligned_sd_b[f"{next_layer_name}.weight"] = torch.matmul(W_next_b, Q.T)

        if return_transformations:
            return aligned_sd_b, rotations
        return aligned_sd_b