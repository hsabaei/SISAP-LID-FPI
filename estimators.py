import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn

class LIDEstimators:
    def __init__(self, device='cpu'):
        self.device = device

    def calculate_Hill(self, V, w):
        """
        Calculate the Hill estimator given arrays V and scalar w.
        """
        if not isinstance(V, np.ndarray):
            raise ValueError("V must be a numpy array")

        # Filter out extremely small values to avoid log(0)
        V_non_zero = V[np.abs(V) != 0]
        k = V_non_zero.shape[0]

        if k == 0:
            return np.nan

        # Compute Hill
        # Add small epsilon to avoid division by zero in log
        epsilon = 1e-7
        Hill = - (k / np.sum(np.log(abs((V_non_zero / (w + 1e-6))))))

        return Hill

    def compute_FIE_LID(self, phi):
        """
        Computes LID using the FIE (Fixed-point Iteration Estimator) approach.
        """
        k = len(phi)

        # w0 and R for numerator
        w0 = phi[0] - phi[k-1]
        R = phi[1:k-2] - phi[k-1]
        Hill_Num = self.calculate_Hill(R, w0)

        # w1 and FR for denominator
        w1 = phi[1] - phi[k-1]
        FR = phi[2:k-1] - phi[k-1]
        Hill_Den = self.calculate_Hill(FR, w1)

        LID_FIE = Hill_Num / Hill_Den
        return LID_FIE

    def compute_GIE_LID(self, phi, G):
        """
        Computes LID using the GIE (General Iteration Estimator) approach.
        """
        k = len(phi)
        # w0 and R for numerator
        w0 = phi[0] - phi[k-1]
        R = phi[1:k-1] - phi[k-1]
        Hill_Num = self.calculate_Hill(R, w0)

        # w1 and FR for denominator
        w1 = G[0] - G[k-1]
        FR = G[1:k-1] - G[k-1]
        Hill_Den = self.calculate_Hill(FR, w1)

        LID_GIE = Hill_Num / Hill_Den
        return LID_GIE

    def compute_Bayesian_LID(self, phi, Num0, Den0):
        """
        Computes LID using a Bayesian-Gamma approach.
        """
        k = len(phi)

        # Denominator part
        w0 = phi[0] - phi[k-1]
        R = phi[1:k-2] - phi[k-1]
        Hill_a = self.calculate_Hill(R, w0)
        Den1 = 1 / Hill_a if Hill_a != 0 else 0

        # Numerator part
        w1 = phi[1] - phi[k-1]
        FR = phi[2:k-1] - phi[k-1]
        Hill_Fa = self.calculate_Hill(FR, w1)
        Num1 = 1 / Hill_Fa if Hill_Fa != 0 else 0

        Num_cumulative = Num0 + Num1
        Den_cumulative = Den0 + Den1

        if Den_cumulative != 0:
            LID_Bay_a = Num_cumulative / Den_cumulative
        else:
            LID_Bay_a = np.nan

        return LID_Bay_a, Num1, Den1

    def compute_IR_LID(self, phi):
        """
        Computes LID using the IR (Iterative Ratio) estimator.
        """
        k = len(phi)
        j = k - 2
        N_diff1 = phi[j+1] - phi[j]
        N_diff2 = phi[j] - phi[j-1]
        D_diff1 = phi[j] - phi[j-1]
        D_diff2 = phi[j-1] - phi[j-2]

        N = np.log(np.abs(N_diff1 / N_diff2))
        D = np.log(np.abs(D_diff1 / D_diff2))
        ratio = N / D
        return ratio

    class Log_Log(nn.Module):
        """
        Inner class for the linear model used in the Log-Log LID estimation.
        """
        def __init__(self):
            super().__init__()
            self.a = nn.Parameter(torch.randn(1, requires_grad=True, dtype=torch.float))
            self.m = nn.Parameter(torch.randn(1, requires_grad=True, dtype=torch.float))

        def forward(self, x):
            return self.m * x + self.a

    def compute_LL_LID(self, X, G, lr=0.01, n_epochs=1000):
        """
        Computes LID using the Log-Log approach.
        """
        k = len(X)
        X = X[1:k-2] - X[k-1]
        X = X[::-1]
        G = G[1:k-2] - G[k-1]
        G = G[::-1]

        GL = np.log(np.abs(G))
        XL = np.log(np.abs(X))

        model = self.Log_Log().to(self.device)
        optimizer = optim.SGD(model.parameters(), lr=lr)
        loss_fn = nn.MSELoss(reduction='mean')

        x_train_tensor = torch.from_numpy(XL).float().to(self.device).view(-1, 1)
        y_train_tensor = torch.from_numpy(GL).float().to(self.device).view(-1, 1)

        for epoch in range(n_epochs):
            model.train()
            optimizer.zero_grad()
            yhat = model(x_train_tensor)
            loss = loss_fn(yhat, y_train_tensor)
            loss.backward()
            optimizer.step()

        return model.m.item()
