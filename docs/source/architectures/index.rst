Full list of architectures (presets)
=====================================================

We list all implemented NF architectures and their respective class names below.
Using these presets facilitates experimentation and modeling, however you can also modify each architecture and build new ones.

.. _autoregressive_architecture_list:

Autoregressive architectures
-----------------------------

We provide the list of autoregressive architectures in the table below.
Import them with ``from torchflows.bijections.finite.autoregressive import <class name>``.
Click the architecture name to see the API and usage examples.
Check the API for all autoregressive architectures :ref:`here <autoregressive_architecture_api>`.

.. list-table::
   :header-rows: 1

   * - Architecture
     - Reference
   * - :class:`NICE <torchflows.bijections.finite.autoregressive.architectures.NICE>`
     - Dinh et al. `NICE: Non-linear Independent Components Estimation <https://arxiv.org/abs/1410.8516>`_ (2015)
   * - :class:`RealNVP <torchflows.bijections.finite.autoregressive.architectures.RealNVP>`
     - Dinh et al. `Density estimation using Real NVP <https://arxiv.org/abs/1605.08803>`_ (2017)
   * - :class:`Inverse RealNVP <torchflows.bijections.finite.autoregressive.architectures.InverseRealNVP>`
     - Dinh et al. `Density estimation using Real NVP <https://arxiv.org/abs/1605.08803>`_ (2017)
   * - :class:`MAF <torchflows.bijections.finite.autoregressive.architectures.MAF>`
     - Papamakarios et al. `Masked Autoregressive Flow for Density Estimation <https://arxiv.org/abs/1705.07057>`_ (2018)
   * - :class:`IAF <torchflows.bijections.finite.autoregressive.architectures.IAF>`
     - Kingma et al. `Improving Variational Inference with Inverse Autoregressive Flow <https://arxiv.org/abs/1606.04934>`_ (2017)
   * - :class:`Coupling RQ-NSF <torchflows.bijections.finite.autoregressive.architectures.CouplingRQNSF>`
     - Durkan et al. `Neural Spline Flows <https://arxiv.org/abs/1906.04032>`_ (2019)
   * - :class:`Masked autoregressive RQ-NSF <torchflows.bijections.finite.autoregressive.architectures.MaskedAutoregressiveRQNSF>`
     - Durkan et al. `Neural Spline Flows <https://arxiv.org/abs/1906.04032>`_ (2019)
   * - :class:`Inverse autoregressive RQ-NSF <torchflows.bijections.finite.autoregressive.architectures.InverseAutoregressiveRQNSF>`
     - Durkan et al. `Neural Spline Flows <https://arxiv.org/abs/1906.04032>`_ (2019)
   * - :class:`Coupling LR-NSF <torchflows.bijections.finite.autoregressive.architectures.CouplingLRS>`
     - Dolatabadi et al. `Invertible Generative Modeling using Linear Rational Splines <https://arxiv.org/abs/2001.05168>`_ (2020)
   * - :class:`Masked autoregressive LR-NSF <torchflows.bijections.finite.autoregressive.architectures.MaskedAutoregressiveLRS>`
     - Dolatabadi et al. `Invertible Generative Modeling using Linear Rational Splines <https://arxiv.org/abs/2001.05168>`_ (2020)
   * - :class:`Inverse autoregressive LR-NSF <torchflows.bijections.finite.autoregressive.architectures.InverseAutoregressiveLRS>`
     - Dolatabadi et al. `Invertible Generative Modeling using Linear Rational Splines <https://arxiv.org/abs/2001.05168>`_ (2020)
   * - :class:`Coupling deep SF <torchflows.bijections.finite.autoregressive.architectures.CouplingDeepSF>`
     - Huang et al. `Neural Autoregressive Flows <https://arxiv.org/abs/1804.00779>`_ (2018)
   * - :class:`Masked autoregressive deep SF <torchflows.bijections.finite.autoregressive.architectures.MaskedAutoregressiveDeepSF>`
     - Huang et al. `Neural Autoregressive Flows <https://arxiv.org/abs/1804.00779>`_ (2018)
   * - :class:`Inverse autoregressive deep SF <torchflows.bijections.finite.autoregressive.architectures.InverseAutoregressiveDeepSF>`
     - Huang et al. `Neural Autoregressive Flows <https://arxiv.org/abs/1804.00779>`_ (2018)
   * - :class:`Coupling dense SF <torchflows.bijections.finite.autoregressive.architectures.CouplingDenseSF>`
     - Huang et al. `Neural Autoregressive Flows <https://arxiv.org/abs/1804.00779>`_ (2018)
   * - :class:`Masked autoregressive dense SF <torchflows.bijections.finite.autoregressive.architectures.MaskedAutoregressiveDenseSF>`
     - Huang et al. `Neural Autoregressive Flows <https://arxiv.org/abs/1804.00779>`_ (2018)
   * - :class:`Inverse autoregressive dense SF <torchflows.bijections.finite.autoregressive.architectures.InverseAutoregressiveDenseSF>`
     - Huang et al. `Neural Autoregressive Flows <https://arxiv.org/abs/1804.00779>`_ (2018)
   * - :class:`Coupling deep-dense SF <torchflows.bijections.finite.autoregressive.architectures.CouplingDeepDenseSF>`
     - Huang et al. `Neural Autoregressive Flows <https://arxiv.org/abs/1804.00779>`_ (2018)
   * - :class:`Masked autoregressive deep-dense SF <torchflows.bijections.finite.autoregressive.architectures.MaskedAutoregressiveDeepDenseSF>`
     - Huang et al. `Neural Autoregressive Flows <https://arxiv.org/abs/1804.00779>`_ (2018)
   * - :class:`Inverse autoregressive deep-dense SF <torchflows.bijections.finite.autoregressive.architectures.InverseAutoregressiveDeepDenseSF>`
     - Huang et al. `Neural Autoregressive Flows <https://arxiv.org/abs/1804.00779>`_ (2018)
   * - :class:`Unconstrained monotonic neural network <torchflows.bijections.finite.autoregressive.architectures.UMNNMAF>`
     - Wehenkel and Louppe `Unconstrained Monotonic Neural Networks <https://arxiv.org/abs/1908.05164>`_ (2019)

.. _multiscale_architecture_list:

Multiscale architectures
-----------------------------------------
We provide the list of multiscale autoregressive architectures in the table below.
Import them with ``from torchflows.bijections.finite.multiscale import <class name>``.
These architectures are specifically made for image modeling, but can also be used for voxels or tensors with more dimensions.
Click the architecture name to see the API and usage examples.
Check the API for all multiscale architectures :ref:`here <multiscale_architecture_api>`.

.. list-table::
   :header-rows: 1

   * - Architecture
     - Reference
   * - :class:`MultiscaleNICE <torchflows.bijections.finite.multiscale.architectures.MultiscaleNICE>`
     - Dinh et al. `NICE: Non-linear Independent Components Estimation <https://arxiv.org/abs/1410.8516>`_ (2015)
   * - :class:`Multiscale RealNVP <torchflows.bijections.finite.multiscale.architectures.MultiscaleRealNVP>`
     - Dinh et al. `Density estimation using Real NVP <https://arxiv.org/abs/1605.08803>`_ (2017)
   * - :class:`Multiscale RQ-NSF <torchflows.bijections.finite.multiscale.architectures.MultiscaleRQNSF>`
     - Durkan et al. `Neural Spline Flows <https://arxiv.org/abs/1906.04032>`_ (2019)
   * - :class:`Multiscale LR-NSF <torchflows.bijections.finite.multiscale.architectures.MultiscaleLRSNSF>`
     - Dolatabadi et al. `Invertible Generative Modeling using Linear Rational Splines <https://arxiv.org/abs/2001.05168>`_ (2020)
   * - :class:`Multiscale deep SF <torchflows.bijections.finite.multiscale.architectures.MultiscaleDeepSigmoid>`
     - Huang et al. `Neural Autoregressive Flows <https://arxiv.org/abs/1804.00779>`_ (2018)
   * - :class:`Multiscale dense SF <torchflows.bijections.finite.multiscale.architectures.MultiscaleDenseSigmoid>`
     - Huang et al. `Neural Autoregressive Flows <https://arxiv.org/abs/1804.00779>`_ (2018)
   * - :class:`Multiscale deep-dense SF <torchflows.bijections.finite.multiscale.architectures.MultiscaleDeepDenseSigmoid>`
     - Huang et al. `Neural Autoregressive Flows <https://arxiv.org/abs/1804.00779>`_ (2018)
   * - :class:`Shift Glow <torchflows.bijections.finite.multiscale.architectures.ShiftGlow>`
     - Kingma and Dhariwal `Glow: Generative Flow with Invertible 1x1 Convolutions <https://arxiv.org/abs/1807.03039>`_ (2018)
   * - :class:`Affine Glow <torchflows.bijections.finite.multiscale.architectures.AffineGlow>`
     - Kingma and Dhariwal `Glow: Generative Flow with Invertible 1x1 Convolutions <https://arxiv.org/abs/1807.03039>`_ (2018)
   * - :class:`RQS Glow <torchflows.bijections.finite.multiscale.architectures.RQSGlow>`
     - Kingma and Dhariwal `Glow: Generative Flow with Invertible 1x1 Convolutions <https://arxiv.org/abs/1807.03039>`_ (2018)
   * - :class:`LRS Glow <torchflows.bijections.finite.multiscale.architectures.LRSGlow>`
     - Kingma and Dhariwal `Glow: Generative Flow with Invertible 1x1 Convolutions <https://arxiv.org/abs/1807.03039>`_ (2018)
   * - :class:`Deep sigmoidal Glow <torchflows.bijections.finite.multiscale.architectures.DeepSigmoidGlow>`
     - Kingma and Dhariwal `Glow: Generative Flow with Invertible 1x1 Convolutions <https://arxiv.org/abs/1807.03039>`_ (2018)
   * - :class:`Dense sigmoidal Glow <torchflows.bijections.finite.multiscale.architectures.DenseSigmoidGlow>`
     - Kingma and Dhariwal `Glow: Generative Flow with Invertible 1x1 Convolutions <https://arxiv.org/abs/1807.03039>`_ (2018)
   * - :class:`Deep-dense sigmoidal Glow <torchflows.bijections.finite.multiscale.architectures.DeepDenseSigmoidGlow>`
     - Kingma and Dhariwal `Glow: Generative Flow with Invertible 1x1 Convolutions <https://arxiv.org/abs/1807.03039>`_ (2018)

.. _residual_architecture_list:

Residual architectures
----------------------------
We provide the list of iterative residual architectures in the table below.
Import them with ``from torchflows.bijections.finite.residual import <class name>``.
Click the architecture name to see the API and usage examples.
Check the API for all residual architectures :ref:`here <residual_architecture_api>`.

.. list-table::
   :header-rows: 1

   * - Architecture
     - Reference
   * - :class:`Invertible ResNet <torchflows.bijections.finite.residual.architectures.InvertibleResNet>`
     - Behrmann et al. `Invertible Residual Networks <https://arxiv.org/abs/1811.00995>`_ (2019)
   * - :class:`ResFlow <torchflows.bijections.finite.residual.architectures.ResFlow>`
     - Chen et al. `Residual Flows for Invertible Generative Modeling <https://arxiv.org/abs/1906.02735>`_ (2020)
   * - :class:`ProximalResFlow <torchflows.bijections.finite.residual.architectures.ProximalResFlow>`
     - Hertrich `Proximal Residual Flows for Bayesian Inverse Problems <https://arxiv.org/abs/2211.17158>`_ (2022)

We also list presets for some convolutional iterative residual architectures in the table below.
These are suitable for image modeling.

.. list-table::
   :header-rows: 1

   * - Architecture
     - Reference
   * - :class:`Convolutional invertible ResNet <torchflows.bijections.finite.residual.architectures.ConvolutionalInvertibleResNet>`
     - Behrmann et al. `Invertible Residual Networks <https://arxiv.org/abs/1811.00995>`_ (2019)
   * - :class:`Convolutional ResFlow <torchflows.bijections.finite.residual.architectures.ConvolutionalResFlow>`
     - Chen et al. `Residual Flows for Invertible Generative Modeling <https://arxiv.org/abs/1906.02735>`_ (2020)

We finally list presets for residual architectures, based on the matrix determinant lemma.
These only implement the inverse transformation.
This means they can be used for sampling (and variational inference), but not for density estimation on arbitrary data.

.. list-table::
   :header-rows: 1

   * - Architecture
     - Reference
   * - :class:`Planar flow <torchflows.bijections.finite.residual.architectures.PlanarBijection>`
     - Rezende and Mohamed `Variational Inference with Normalizing Flows <https://arxiv.org/abs/1505.05770>`_ (2016)
   * - :class:`Radial flow <torchflows.bijections.finite.residual.architectures.RadialBijection>`
     - Rezende and Mohamed `Variational Inference with Normalizing Flows <https://arxiv.org/abs/1505.05770>`_ (2016)
   * - :class:`Sylvester flow <torchflows.bijections.finite.residual.architectures.SylvesterBijection>`
     - Van den Berg et al. `Sylvester Normalizing Flows for Variational Inference <https://arxiv.org/abs/1803.05649>`_ (2019)

.. _continuous_architecture_list:

Continuous architectures
----------------------------
We provide the list of continuous architectures in the table below.
Import them with ``from torchflows.bijections.continuous import <class name>`` and wrap them with :class:`~torchflows.flows.ContinuousFlow`.
Click the architecture name to see the API and usage examples.
Check the API for all continuous architectures :ref:`here <continuous_architecture_api>`.

.. list-table::
   :header-rows: 1

   * - Architecture
     - Reference
   * - :class:`DDNF <torchflows.bijections.continuous.ddnf.DDNF>`
     - Salman et al. `Deep Diffeomorphic Normalizing Flows <https://arxiv.org/abs/1810.03256>`_ (2018)
   * - :class:`FFJORD <torchflows.bijections.continuous.ffjord.FFJORD>`
     - Grathwohl et al. `FFJORD: Free-form Continuous Dynamics for Scalable Reversible Generative Models <https://arxiv.org/abs/1810.01367>`_ (2018)
   * - :class:`RNODE <torchflows.bijections.continuous.rnode.RNODE>`
     - Finlay et al. `How to train your neural ODE: the world of Jacobian and kinetic regularization <https://arxiv.org/abs/2002.02798>`_ (2020)
   * - :class:`OT-Flow <torchflows.bijections.continuous.otflow.OTFlowBijection>`
     - Onken et al. `OT-Flow: Fast and Accurate Continuous Normalizing Flows via Optimal Transport <https://arxiv.org/abs/2006.00104>`_ (2021)

We also list presets for convolutional continuous architectures in the table below.
These are suitable for image modeling.

.. list-table::
   :header-rows: 1

   * - Architecture
     - Reference
   * - :class:`Convolutional DDNF <torchflows.bijections.continuous.ddnf.ConvolutionalDDNF>`
     - Salman et al. `Deep Diffeomorphic Normalizing Flows <https://arxiv.org/abs/1810.03256>`_ (2018)
   * - :class:`Convolutional FFJORD <torchflows.bijections.continuous.ffjord.ConvolutionalFFJORD>`
     - Grathwohl et al. `FFJORD: Free-form Continuous Dynamics for Scalable Reversible Generative Models <https://arxiv.org/abs/1810.01367>`_ (2018)
   * - :class:`Convolutional RNODE <torchflows.bijections.continuous.rnode.ConvolutionalRNODE>`
     - Finlay et al. `How to train your neural ODE: the world of Jacobian and kinetic regularization <https://arxiv.org/abs/2002.02798>`_ (2020)
