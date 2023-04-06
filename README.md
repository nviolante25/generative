# Implementation of generative models

- [VAE](vae.py) from ["Auto-Encoding Variational Bayes"](https://arxiv.org/abs/1312.6114) Diederik P. Kingma, Max Welling.

    <div>
        <div style="display: flex; justify-content: left;">
        <figure>
        <figcaption>Frey Faces 28x20</figcaption>
        <img src="./images/vae_frey_faces.png">
        </figure>
    </div>

- [VQ-VAE](vq_vae.py) from ["Neural Discrete Representation Learning"](https://arxiv.org/abs/1711.00937) Aaron van den Oord, Oriol Vinyals, Koray Kavukcuoglu.

    <div style="display: flex; justify-content: left;">
        <figure>
        <img src="./images/vq_vae.png">
        <figcaption>Original</figcaption>
        </figure>
        <figure>
        <img src="./images/vq_vae_rec.png">
        <figcaption>Reconstruction</figcaption>
        </figure>
    </div>


- [ProgressiveGAN](progan.py) from ["Progressive Growing of GANs for Improved Quality, Stability, and Variation"](https://arxiv.org/abs/1710.10196) Tero Karras, Timo Aila, Samuli Laine, Jaakko Lehtinen


    <div style="display: flex; justify-content: left;">
        <figure>
        <figcaption>CIFAR10 32x32</figcaption>
        <img src="./images/progan_cifar10.gif">
        </figure>
    </div>
