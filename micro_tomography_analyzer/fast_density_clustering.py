import numpy as np
import scipy.stats as st

class FastDensityClustering():
    @staticmethod
    def gaussian_kernel(size=21, nsig=3):
        """Returns a 2D Gaussian kernel.
        Args:
            size: The size of the kernel (size x size)
            nsig: Sigma of the gaussian
        """
        x = np.linspace(-nsig, nsig, size+1)
        kern1d = np.diff(st.norm.cdf(x))
        kern2d = np.outer(kern1d, kern1d)
        return kern2d/kern2d.sum()

    @staticmethod
    def kernel(size, ktype):
        """ Returns a kernel of specified size and type
        Args:
            size: Kernel size
            ktype: Type of kernel. Either uniform gaussian or disk are provided.
        """
        if ktype == "uniform":
            return np.ones((size,size))
        elif ktype == "gaussian":
            k = FastDensityClustering.gaussian_kernel(size=size)
            k /= np.max(k)
            return k
        elif ktype == "disk":
            k = FastDensityClustering.gaussian_kernel(size=size)
            k /= np.max(k)
            return k > 0.03

    @staticmethod
    def collapse_iteration(arr,kernel):
        """ Determins center of gravity for each non-zero (forground pixel) and it's surround weighted by the kernel
            and increases mass at named target position/pixel by the mass of the source pixel.
        Args:
            arr: Grayscale array of positive values where value zero stands for the background and positive values denote the mass for a given foreground pixel.
            kernel: Kernel used to weight the influance of nearby pixels in computing the center of mass
        """
        kernel_width = kernel.shape[0]
        kernel_height = kernel.shape[1]
        ys, xs = np.where(arr>0)
        new = np.zeros(arr.shape)
        abs_shift = 0

        mapping = {}
        for y, x in zip(ys,xs):
            snippet = arr[y-kernel_width//2:(y+kernel_width//2)+1, x-kernel_width//2:(x+kernel_width//2)+1]

            snippet = kernel * snippet
            weights_x = np.mean(snippet,axis=0)
            weights_y = np.mean(snippet,axis=1)

            shift_x = np.average(np.arange(kernel_width),weights=weights_x)#The inner mean returns x values, the outer is their mean -> shift x
            shift_y = np.average(np.arange(kernel_height),weights=weights_y)#The inner mean returns y values, the outer is their mean -> shift y
            shift_x -= (kernel_width-1)/2
            shift_y -= (kernel_height-1)/2


            y1 = int(y+shift_y)
            x1 = int(x+shift_x)

            #Remember where the contribution of the mass of the tatget comes from
            if y1 != y or x1 != x:
                if not str([y1,x1]) in mapping:
                    mapping[str([y1,x1])] = []
                mapping[str([y1,x1])] = [y,x]

            abs_shift += np.abs(shift_x) + np.abs(shift_y)
            new[y1,x1] += arr[y,x]

        return new, abs_shift/len(xs), mapping

    @staticmethod
    def collapse(arr, iterations = None,gravity_type="uniform", gravity_size=5):
        """ Performs clustering by iteratively moving all mass densities (non-zero/foreground pixels) to their center of mass.
        If no value for iterations is specified the algorithm runs until convergence is achieved and the movement is marginally.
        Args:
            arr: Array of positive gray values
            iterations: Number of iterations. If no value for iterations is specified the algorithm runs until convergence is achieved.
            gravity_type: Either "uniform", "gaussian" or "disk". The contributions to the center of mass for one pixels by its surround are weighted accordingly.
            gravity_size: The size of the gravity kernel.
        Returns:
            Array representation of cluster centers. Each cluster center is represented by a non-zero pixel.
        """
        epsilon = None
        if not iterations:
            iterations = 1000
            epsilon = 1.0e-16

        if gravity_size % 2 == 0:
            gravity_size += 1
        k = FastDensityClustering.kernel(gravity_size,gravity_type)
        arr = np.pad(arr,gravity_size, "constant")
        mappings = []
        for x in range(iterations):
            print(".",end="")
            arr, shift, mapping = FastDensityClustering.collapse_iteration(arr,k)
            mappings.append(mapping)
            if epsilon:
                if epsilon > shift:
                    break


        return arr[gravity_size:-gravity_size,gravity_size:-gravity_size], mappings

    @staticmethod
    def density_clustering(arr, iterations = None,gravity_type="uniform", gravity_size=5):
        """ Performs clustering by iteratively moving all mass densities (non-zero/foreground pixels) to their center of mass.
        If no value for iterations is specified the algorithm runs until convergence is achieved and the movement is marginally.
        Args:
            arr: Array of positive gray values
            iterations: Number of iterations. If no value for iterations is specified the algorithm runs until convergence is achieved.
            gravity_type: Either "uniform", "gaussian" or "disk". The contributions to the center of mass for one pixels by its surround are weighted accordingly.
            gravity_size: The size of the gravity kernel.
        Returns:
            Y and x positions of all detected cluster centers
        """
        cluster_array, mappings = FastDensityClustering.collapse(arr,iterations,gravity_type=gravity_type, gravity_size=gravity_size)
        center_y,center_x = np.where(cluster_array>0)
        return center_y, center_x, cluster_array, mappings
