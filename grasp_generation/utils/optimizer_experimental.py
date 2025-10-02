"""
Last modified date: 2025.09.28
Author: you
Description: Class optimizer (leaf-param version: rotation & joint_angles only)
"""

import torch


class ExperimentalAnnealing:
    def __init__(
        self,
        hand_model,
        switch_possibility=0.1,
        starting_temperature=3.0,
        temperature_decay=0.95,
        annealing_period=10,
        noise_size=0.05,
        stepsize_period=10,
        mu=0.98,
        device="cpu",
        optimize_translation=False,  # if True, add translation as a leaf param too
    ):
        self.hand_model = hand_model
        self.device = device
        self.optimize_translation = optimize_translation

        # hyperparams / schedules
        self.switch_possibility = switch_possibility
        self.starting_temperature = torch.tensor(starting_temperature, dtype=torch.float, device=device)
        self.temperature_decay   = torch.tensor(temperature_decay,   dtype=torch.float, device=device)
        self.annealing_period    = torch.tensor(annealing_period,    dtype=torch.long,  device=device)
        self.noise_size          = torch.tensor(noise_size,          dtype=torch.float, device=device)
        self.step_size_period    = torch.tensor(stepsize_period,     dtype=torch.long,  device=device)
        self.mu                  = torch.tensor(mu,                  dtype=torch.float, device=device)
        self.step = 0

        # Old state for accept/reject
        self.old_rotation = None           # (B, 3,3) or (B,6) depending on your representation
        self.old_joint_angles = None       # (B, DoF)
        self.old_contact_point_indices = None

        # EMA buffers (match shapes of leaf params)
        self.ema_rot  = torch.zeros_like(self.hand_model.rotation,      dtype=torch.float, device=device)
        self.ema_jnts = torch.zeros_like(self.hand_model.joint_angles,  dtype=torch.float, device=device)

        # If you ever decide to optimize translation too, uncomment these:
        self.ema_T = torch.zeros_like(self.hand_model.translation, dtype=torch.float, device=device)

    @torch.no_grad()
    def _adaptive_step(self, param, ema, step_scale):
        """
        RMSProp-style update:  p <- p - step_scale * grad / (sqrt(ema) + eps)
        Expects param.grad to be populated (call after loss.backward()).
        """
        if param.grad is None:
            return  # nothing to do (frozen or no gradient this iter)

        # update EMA of squared grads
        ema.mul_(self.mu).addcmul_(param.grad, param.grad, value=(1.0 - float(self.mu)))

        # scale step (broadcast scalar step_scale to param shape)
        denom = ema.sqrt().add_(1e-6)
        param.addcdiv_(param.grad, denom, value=-float(step_scale))

    def try_step(self):
        """
        Propose a step:
        - Uses grads on leaf params: rotation & joint_angles (and translation if you want)
        - Randomly switches a subset of contact points
        - Calls hand_model.set_parameters(...) with updated leaf params
        NOTE: Call this *after* loss.backward() each iteration.
        """
        # scalar step size with annealing
        s = float(self.noise_size * (self.temperature_decay ** torch.div(self.step, self.step_size_period, rounding_mode="floor")))

        # Save old state (for accept/reject)
        self.old_rotation = self.hand_model.rotation.detach().clone()
        self.old_joint_angles = self.hand_model.joint_angles.detach().clone()
        self.old_contact_point_indices = self.hand_model.contact_point_indices.detach().clone()

        # Update leaf params with grads (RMSProp-style)
        with torch.no_grad():
            self._adaptive_step(self.hand_model.rotation,      self.ema_rot,  s)
            self._adaptive_step(self.hand_model.joint_angles,  self.ema_jnts, s)
            # If optimizing translation too:
            if self.optimize_translation:
                self._adaptive_step(self.hand_model.translation, self.ema_T, s)

        # Randomly switch a subset of contacts
        batch_size, n_contact = self.hand_model.contact_point_indices.shape
        switch_mask = torch.rand(batch_size, n_contact, dtype=torch.float, device=self.device) < self.switch_possibility
        contact_point_indices = self.hand_model.contact_point_indices.clone()
        if switch_mask.any():
            contact_point_indices[switch_mask] = torch.randint(
                self.hand_model.n_contact_candidates,
                size=(int(switch_mask.sum().item()),),
                device=self.device
            )

        # Re-pack in the hand model (you said you'll build hand_pose there)
        # IMPORTANT: Do not detach here; keep the latest leaf params.
        params = {
            "translation": self.hand_model.translation if self.optimize_translation else self.hand_model.translation.detach(),
            "rotation": self.hand_model.rotation,
            "joint_angles": self.hand_model.joint_angles,
            "contact_point_indices": contact_point_indices,
        }
        self.hand_model.set_parameters(**params)

        self.step += 1
        return s

    @torch.no_grad()
    def accept_step(self, energy, new_energy):
        """
        Metropolis accept/reject. If rejected, restore leaf params & contact indices.
        """
        batch_size = energy.shape[0]
        temperature = self.starting_temperature * self.temperature_decay ** torch.div(
            self.step, self.annealing_period, rounding_mode="floor"
        )

        alpha = torch.rand(batch_size, dtype=torch.float, device=self.device)
        accept = alpha < torch.exp((energy - new_energy) / temperature)

        # Restore rejected batch entries for leaf params and contact indices
        reject = ~accept
        if reject.any():
            # rotation may be (B,3,3) or (B,6); indexing works the same
            self.hand_model.rotation[reject] = self.old_rotation[reject]
            self.hand_model.joint_angles[reject] = self.old_joint_angles[reject]
            self.hand_model.contact_point_indices[reject] = self.old_contact_point_indices[reject]

            # Re-pack so downstream caches (FK, contacts) are consistent
            params = {
                "translation": self.hand_model.translation if self.optimize_translation else self.hand_model.translation.detach(),
                "rotation": self.hand_model.rotation,
                "joint_angles": self.hand_model.joint_angles,
                "contact_point_indices": self.hand_model.contact_point_indices,
            }
            self.hand_model.set_parameters(**params)

        return accept, temperature

    @torch.no_grad()
    def zero_grad(self):
        """
        Zero grads on leaf Parameters (rotation & joints; add translation if trainable).
        """
        if self.hand_model.rotation.grad is not None:
            self.hand_model.rotation.grad.zero_()
        if self.hand_model.joint_angles.grad is not None:
            self.hand_model.joint_angles.grad.zero_()
        if self.optimize_translation and hasattr(self.hand_model, "translation") and self.hand_model.translation.requires_grad:
            if self.hand_model.translation.grad is not None:
                self.hand_model.translation.grad.zero_()
