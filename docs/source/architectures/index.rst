Full list of architectures (presets)
=====================================================

We list all implemented NF architectures and their respective class names below.
Using these presets facilitates experimentation and modeling, however you can also modify each architecture and build new ones.

.. _autoregressive_architecture_list:

Autoregressive architectures
-----------------------------

We provide the list of autoregressive architectures in the table below.
Click the architecture name to see the API and usage examples.
Check the API for all autoregressive architectures :ref:`here <autoregressive_architecture_api>`.

.. list-table::
   :header-rows: 1

   * - Architecture
     - Reference
   * - :class:`NICE <torchflows.bijections.finite.autoregressive.architectures.NICE>`
     - Dinh et al. `NICE: Non-linear Independent Components Estimation <http://arxiv.org/abs/1410.8516>`_ (2015)
   * - :class:`RealNVP <torchflows.bijections.finite.autoregressive.architectures.RealNVP>`
     - Dinh et al. `Density estimation using Real NVP <http://arxiv.org/abs/1605.08803>`_ (2017)
   * - :class:`Inverse RealNVP <torchflows.bijections.finite.autoregressive.architectures.InverseRealNVP>`
     - Dinh et al. `Density estimation using Real NVP <http://arxiv.org/abs/1605.08803>`_ (2017)
   * - :class:`MAF <torchflows.bijections.finite.autoregressive.architectures.MAF>`
     - Papamakarios et al. `Masked Autoregressive Flow for Density Estimation <http://arxiv.org/abs/1705.07057>`_ (2018)
   * - :class:`IAF <torchflows.bijections.finite.autoregressive.architectures.IAF>`
     - Kingma et al. `Improving Variational Inference with Inverse Autoregressive Flow <http://arxiv.org/abs/1606.04934>`_ (2017)
   * - :class:`Coupling RQ-NSF <torchflows.bijections.finite.autoregressive.architectures.CouplingRQNSF>`
     - Durkan et al. `Neural Spline Flows <http://arxiv.org/abs/1906.04032>`_ (2019)
   * - :class:`Masked autoregressive RQ-NSF <torchflows.bijections.finite.autoregressive.architectures.MaskedAutoregressiveRQNSF>`
     - Durkan et al. `Neural Spline Flows <http://arxiv.org/abs/1906.04032>`_ (2019)
   * - :class:`Inverse autoregressive RQ-NSF <torchflows.bijections.finite.autoregressive.architectures.InverseAutoregressiveRQNSF>`
     - Durkan et al. `Neural Spline Flows <http://arxiv.org/abs/1906.04032>`_ (2019)
   * - :class:`Coupling LR-NSF <torchflows.bijections.finite.autoregressive.architectures.CouplingLRS>`
     - Dolatabadi et al. `Invertible Generative Modeling using Linear Rational Splines <http://arxiv.org/abs/2001.05168>`_ (2020)
   * - :class:`Masked autoregressive LR-NSF <torchflows.bijections.finite.autoregressive.architectures.MaskedAutoregressiveLRS>`
     - Dolatabadi et al. `Invertible Generative Modeling using Linear Rational Splines <http://arxiv.org/abs/2001.05168>`_ (2020)
   * - :class:`Inverse autoregressive LR-NSF <torchflows.bijections.finite.autoregressive.architectures.InverseAutoregressiveLRS>`
     - Dolatabadi et al. `Invertible Generative Modeling using Linear Rational Splines <http://arxiv.org/abs/2001.05168>`_ (2020)
   * - :class:`Coupling deep SF <torchflows.bijections.finite.autoregressive.architectures.CouplingDeepSF>`
     -
   * - :class:`Masked autoregressive deep SF <torchflows.bijections.finite.autoregressive.architectures.MaskedAutoregressiveDeepSF>`
     -
   * - :class:`Inverse autoregressive deep SF <torchflows.bijections.finite.autoregressive.architectures.InverseAutoregressiveDeepSF>`
     -
   * - :class:`Coupling dense SF <torchflows.bijections.finite.autoregressive.architectures.CouplingDenseSF>`
     -
   * - :class:`Masked autoregressive dense SF <torchflows.bijections.finite.autoregressive.architectures.MaskedAutoregressiveDenseSF>`
     -
   * - :class:`Inverse autoregressive dense SF <torchflows.bijections.finite.autoregressive.architectures.InverseAutoregressiveDenseSF>`
     -
   * - :class:`Coupling deep-dense SF <torchflows.bijections.finite.autoregressive.architectures.CouplingDeepDenseSF>`
     -
   * - :class:`Masked autoregressive deep-dense SF <torchflows.bijections.finite.autoregressive.architectures.MaskedAutoregressiveDeepDenseSF>`
     -
   * - :class:`Inverse autoregressive deep-dense SF <torchflows.bijections.finite.autoregressive.architectures.InverseAutoregressiveDeepDenseSF>`
     -
   * - :class:`Unconstrained monotonic neural network <torchflows.bijections.finite.autoregressive.architectures.UMNNMAF>`
     -

.. _multiscale_architecture_list:

Multiscale architectures
-----------------------------------------
We provide the list of multiscale autoregressive architectures in the table below.
These architectures are specifically made for image modeling, but can also be used for voxels or tensors with more dimensions.
Click the architecture name to see the API and usage examples.
Check the API for all multiscale architectures :ref:`here <multiscale_architecture_api>`.

.. list-table::
   :header-rows: 1

   * - Architecture
     - Reference
   * - :class:`MultiscaleNICE <torchflows.bijections.finite.multiscale.architectures.MultiscaleNICE>`
     - Dinh et al. `NICE: Non-linear Independent Components Estimation <http://arxiv.org/abs/1410.8516>`_ (2015)
   * - :class:`Multiscale RealNVP <torchflows.bijections.finite.multiscale.architectures.MultiscaleRealNVP>`
     - Dinh et al. `Density estimation using Real NVP <http://arxiv.org/abs/1605.08803>`_ (2017)
   * - :class:`Multiscale RQ-NSF <torchflows.bijections.finite.multiscale.architectures.MultiscaleRQNSF>`
     - Durkan et al. `Neural Spline Flows <http://arxiv.org/abs/1906.04032>`_ (2019)
   * - :class:`Multiscale LR-NSF <torchflows.bijections.finite.multiscale.architectures.MultiscaleLRSNSF>`
     - Dolatabadi et al. `Invertible Generative Modeling using Linear Rational Splines <http://arxiv.org/abs/2001.05168>`_ (2020)
   * - :class:`Multiscale deep SF <torchflows.bijections.finite.multiscale.architectures.MultiscaleDeepSigmoid>`
     -
   * - :class:`Multiscale dense SF <torchflows.bijections.finite.multiscale.architectures.MultiscaleDenseSigmoid>`
     -
   * - :class:`Multiscale deep-dense SF <torchflows.bijections.finite.multiscale.architectures.MultiscaleDeepDenseSigmoid>`
     -
   * - :class:`Shift Glow <torchflows.bijections.finite.multiscale.architectures.ShiftGlow>`
     -
   * - :class:`Affine Glow <torchflows.bijections.finite.multiscale.architectures.AffineGlow>`
     -
   * - :class:`RQS Glow <torchflows.bijections.finite.multiscale.architectures.RQSGlow>`
     -
   * - :class:`LRS Glow <torchflows.bijections.finite.multiscale.architectures.LRSGlow>`
     -
   * - :class:`Deep sigmoidal Glow <torchflows.bijections.finite.multiscale.architectures.DeepSigmoidGlow>`
     -
   * - :class:`Dense sigmoidal Glow <torchflows.bijections.finite.multiscale.architectures.DenseSigmoidGlow>`
     -
   * - :class:`Deep-dense sigmoidal Glow <torchflows.bijections.finite.multiscale.architectures.DeepDenseSigmoidGlow>`
     -

Residual architectures
----------------------------
We provide the list of iterative residual architectures in the table below.
Click the architecture name to see the API and usage examples.
Check the API for all residual architectures :ref:`here <residual_architecture_api>`.

.. list-table::
   :header-rows: 1

   * - Architecture
     - Reference
   * - :class:`Invertible ResNet <torchflows.bijections.finite.residual.architectures.InvertibleResNet>`
     -
   * - :class:`ResFlow <torchflows.bijections.finite.residual.architectures.ResFlow>`
     -
   * - :class:`ProximalResFlow <torchflows.bijections.finite.residual.architectures.ProximalResFlow>`
     -

We also list presets for some convolutional iterative residual architectures in the table below.
These are suitable for image modeling.

.. list-table::
   :header-rows: 1

   * - Architecture
     - Reference
   * - :class:`Convolutional invertible ResNet <torchflows.bijections.finite.residual.architectures.ConvolutionalInvertibleResNet>`
     -
   * - :class:`Convolutional ResFlow <torchflows.bijections.finite.residual.architectures.ConvolutionalResFlow>`
     -

We finally list presets for residual architectures, based on the matrix determinant lemma.
These support either forward or inverse transformation, but not both.
This means they can be used for either sampling (and variational inference) or density estimation (and maximum likelihood fits), but not both at the same time.

.. list-table::
   :header-rows: 1

   * - Architecture
     - Reference
   * - :class:`Planar flow <torchflows.bijections.finite.residual.architectures.PlanarFlow>`
     -
   * - :class:`Radial flow <torchflows.bijections.finite.residual.architectures.RadialFlow>`
     -
   * - :class:`Sylvester flow <torchflows.bijections.finite.residual.architectures.SylvesterFlow>`
     -

Continuous architectures
----------------------------
We provide the list of continuous architectures in the table below.
Click the architecture name to see the API and usage examples.
Check the API for all continuous architectures :ref:`here <continuous_architecture_api>`.

.. list-table::
   :header-rows: 1

   * - Architecture
     - Reference
   * - :class:`DDNF <torchflows.bijections.finite.continuous.ddnf.DeepDiffeomorphicBijection>`
     -
   * - :class:`FFJORD <torchflows.bijections.finite.continuous.ffjord.FFJORD>`
     -
   * - :class:`RNODE <torchflows.bijections.finite.continuous.rnode.RNODE>`
     -
   * - :class:`OT-Flow <torchflows.bijections.finite.continuous.otflow.OTFlow>`
     -

We also list presets for convolutional continuous architectures in the table below.
These are suitable for image modeling.

.. list-table::
   :header-rows: 1

   * - Architecture
     - Reference
   * - :class:`Convolutional DDNF <torchflows.bijections.finite.continuous.ddnf.ConvolutionalDeepDiffeomorphicBijection>`
     -
   * - :class:`Convolutional FFJORD <torchflows.bijections.finite.continuous.ffjord.ConvolutionalFFJORD>`
     -
   * - :class:`Convolutional RNODE <torchflows.bijections.finite.continuous.rnode.ConvolutionalRNODE>`
     -