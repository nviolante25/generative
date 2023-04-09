# Implementation of generative models

- [VAE](vae.py) from ["Auto-Encoding Variational Bayes"](https://arxiv.org/abs/1312.6114) Diederik P. Kingma, Max Welling.

    ![VAE_frey_faces](./images/vae_frey_faces.png)

- [VQ-VAE](vq_vae.py) from ["Neural Discrete Representation Learning"](https://arxiv.org/abs/1711.00937) Aaron van den Oord, Oriol Vinyals, Koray Kavukcuoglu.

    ![Original_VQVAE](./images/vq_vae.png)
    ![Rec_VQVAE](./images/vq_vae_rec.png)


- [ProGAN](progan.py) from ["Progressive Growing of GANs for Improved Quality, Stability, and Variation"](https://arxiv.org/abs/1710.10196) Tero Karras, Timo Aila, Samuli Laine, Jaakko Lehtinen

    <table>
        <tr>
            <th>CIFAR10 32x32</th>
            <th>AFHQ 256x256</th>
        </tr>
        <tr>
            <td><img src="./images/progan_cifar10_32x32.gif"></td>
            <td><img src="./images/progan_afhq_256x256.gif"></td>
        </tr>
    </table>
