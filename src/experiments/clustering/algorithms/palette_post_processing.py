#different post processing implementations for palette

from utils.distance_metrics import tam_euclidian_distance
from utils.trace_operations import compute_super_matrix
from tqdm import tqdm
def refine_clusters_with_swaps_heavy(tams, S_A, S_M, max_iterations=50):
    """
    Refine clustering by swapping tams between clusters when beneficial.
    
    Args:
        tams: List of tam instances (2*n numpy arrays)
        S_A: List of lists containing indices of tams in each cluster
        S_M: List of super matrices for each cluster
        max_iterations: Maximum number of refinement iterations
    
    Returns:
        Tuple of (refined S_A, refined S_M)
    """

    """
    Instead of moving tams, it looks for beneficial swap opportunities between clusters
    For each tam, it checks all possible swaps with tams in other clusters
    A swap is only performed if it reduces the total distance (sum of distances of both tams to their new clusters)
    Cluster sizes remain constant throughout the process since we're only swapping tams
    After each swap, it updates the super matrices of both affected clusters
    """
    num_clusters = len(S_A)
    
    
    def calculate_swap_benefit(tam1_idx, tam2_idx, cluster1_idx, cluster2_idx):
        """Calculate if swapping two tams between clusters would reduce total distance."""
        tam1 = tams[tam1_idx]
        tam2 = tams[tam2_idx]
        
        # Calculate current distances
        current_distance1 = tam_euclidian_distance(tam1, S_M[cluster1_idx])
        current_distance2 = tam_euclidian_distance(tam2, S_M[cluster2_idx])
        current_total = current_distance1 + current_distance2
        
        # Temporarily remove tams from their clusters
        cluster1_tams = [tams[idx] for idx in S_A[cluster1_idx] if idx != tam1_idx]
        cluster2_tams = [tams[idx] for idx in S_A[cluster2_idx] if idx != tam2_idx]
        
        # Add tams to opposite clusters
        new_cluster1_tams = cluster1_tams + [tam2]
        new_cluster2_tams = cluster2_tams + [tam1]
        
        # Compute new super matrices
        new_S_M1 = compute_super_matrix(*new_cluster1_tams)
        new_S_M2 = compute_super_matrix(*new_cluster2_tams)
        
        # Calculate new distances
        new_distance1 = tam_euclidian_distance(tam2, new_S_M1)
        new_distance2 = tam_euclidian_distance(tam1, new_S_M2)
        new_total = new_distance1 + new_distance2
        
        # Return improvement in total distance
        return current_total - new_total
    
    
    number_of_swaps = 0
    for iteration in tqdm(range(max_iterations),desc= 'performing post processing with n^2 swaps'):
        changes_made = False
        
        # For each cluster
        for cluster1_idx in range(num_clusters):
            # For each tam in current cluster
            for tam1_pos, tam1_idx in enumerate(S_A[cluster1_idx]):
                best_improvement = 0
                best_swap = None
                
                # Check potential swaps with tams in other clusters
                for cluster2_idx in range(num_clusters):
                    if cluster2_idx == cluster1_idx:
                        continue
                        
                    for tam2_pos, tam2_idx in enumerate(S_A[cluster2_idx]):
                        improvement = calculate_swap_benefit(
                            tam1_idx, tam2_idx,
                            cluster1_idx, cluster2_idx
                        )
                        
                        if improvement > best_improvement:
                            best_improvement = improvement
                            best_swap = (cluster2_idx, tam2_pos, tam2_idx)
                
                # If beneficial swap found, perform it
                if best_swap and best_improvement > 0:
                    number_of_swaps += 1
                    changes_made = True
                    cluster2_idx, tam2_pos, tam2_idx = best_swap
                    
                    # Swap tams between clusters
                    S_A[cluster1_idx][tam1_pos], S_A[cluster2_idx][tam2_pos] = \
                        S_A[cluster2_idx][tam2_pos], S_A[cluster1_idx][tam1_pos]
                    
                    # Update super matrices for both clusters
                    cluster1_tams = [tams[idx] for idx in S_A[cluster1_idx]]
                    cluster2_tams = [tams[idx] for idx in S_A[cluster2_idx]]
                    
                    S_M[cluster1_idx] = compute_super_matrix(*cluster1_tams)
                    S_M[cluster2_idx] = compute_super_matrix(*cluster2_tams)
        
        if not changes_made:
            print(f"Converged after {iteration} iterations")
            break
    print(f'{number_of_swaps} of swaps were performed')
    if iteration == max_iterations:
        print(f"Reached maximum iterations ({max_iterations}) without convergence")
    
    return S_A, S_M


def refine_clusters_with_directed_swaps_medium(tams, S_A, S_M, max_iterations=20):
    """
    Refine clustering by first finding the closest cluster for each tam, 
    then finding the best swap partner in that cluster.
    
    Args:
        tams: List of tam instances (2*n numpy arrays)
        S_A: List of lists containing indices of tams in each cluster
        S_M: List of super matrices for each cluster
        tam_euclidian_distance: Function to compute distance between two tams
        compute_super_matrix: Function to compute super matrix from multiple tams
        max_iterations: Maximum number of refinement iterations
    
    Returns:
        Tuple of (refined S_A, refined S_M)
    """
    num_clusters = len(S_A)
    
    
    def find_closest_cluster(tam_idx, current_cluster_idx):
        """Find the cluster with the smallest distance to the given tam."""
        tam = tams[tam_idx]
        min_distance = float('inf')
        best_cluster_idx = None
        
        for cluster_idx in range(num_clusters):
            if cluster_idx == current_cluster_idx:
                continue
            
            distance = tam_euclidian_distance(tam, S_M[cluster_idx])
            if distance < min_distance:
                min_distance = distance
                best_cluster_idx = cluster_idx
                
        return best_cluster_idx
    
    def calculate_swap_improvement(tam1_idx, tam2_idx, cluster1_idx, cluster2_idx):
        """Calculate improvement in total distance after swapping two tams."""
        tam1 = tams[tam1_idx]
        tam2 = tams[tam2_idx]
        
        # Calculate current distances
        current_distance1 = tam_euclidian_distance(tam1, S_M[cluster1_idx])
        current_distance2 = tam_euclidian_distance(tam2, S_M[cluster2_idx])
        current_total = current_distance1 + current_distance2
        
        # Temporarily remove tams from their clusters
        cluster1_tams = [tams[idx] for idx in S_A[cluster1_idx] if idx != tam1_idx]
        cluster2_tams = [tams[idx] for idx in S_A[cluster2_idx] if idx != tam2_idx]
        
        # Add tams to opposite clusters
        new_cluster1_tams = cluster1_tams + [tam2]
        new_cluster2_tams = cluster2_tams + [tam1]
        
        # Compute new super matrices
        new_S_M1 = compute_super_matrix(*new_cluster1_tams)
        new_S_M2 = compute_super_matrix(*new_cluster2_tams)
        
        # Calculate new distances
        new_distance1 = tam_euclidian_distance(tam2, new_S_M1)
        new_distance2 = tam_euclidian_distance(tam1, new_S_M2)
        new_total = new_distance1 + new_distance2
        
        return current_total - new_total
    
    
    number_of_swaps = 0

    for iteration in tqdm(range(max_iterations),desc= 'performing lighter post processing'):

        changes_made = False
        # For each cluster
        for cluster1_idx in range(num_clusters):
            # For each tam in current cluster
            for tam1_pos, tam1_idx in enumerate(S_A[cluster1_idx]):
                # Find the closest cluster for this tam
                closest_cluster_idx = find_closest_cluster(tam1_idx, cluster1_idx)
                
                if closest_cluster_idx is None:
                    continue
                
                # Find the best swap partner in the closest cluster
                best_improvement = 0
                best_swap_pos = None
                best_swap_idx = None
                
                for tam2_pos, tam2_idx in enumerate(S_A[closest_cluster_idx]):
                    improvement = calculate_swap_improvement(
                        tam1_idx, tam2_idx,
                        cluster1_idx, closest_cluster_idx
                    )
                    
                    if improvement > best_improvement:
                        best_improvement = improvement
                        best_swap_pos = tam2_pos
                        best_swap_idx = tam2_idx
                
                # If beneficial swap found, perform it
                if best_swap_pos is not None and best_improvement > 0:
                    number_of_swaps += 1
                    changes_made = True
                    
                    # Swap tams between clusters
                    S_A[cluster1_idx][tam1_pos], S_A[closest_cluster_idx][best_swap_pos] = \
                        S_A[closest_cluster_idx][best_swap_pos], S_A[cluster1_idx][tam1_pos]
                    
                    # Update super matrices for both clusters
                    cluster1_tams = [tams[idx] for idx in S_A[cluster1_idx]]
                    cluster2_tams = [tams[idx] for idx in S_A[closest_cluster_idx]]
                    
                    S_M[cluster1_idx] = compute_super_matrix(*cluster1_tams)
                    S_M[closest_cluster_idx] = compute_super_matrix(*cluster2_tams)
        
        if not changes_made:
            print(f"Converged after {iteration} iterations")
            break
    print(f'{number_of_swaps} of swaps were performed')
    if iteration == max_iterations:
        print(f"Reached maximum iterations ({max_iterations}) without convergence")
    

    
    return S_A, S_M