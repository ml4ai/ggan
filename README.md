Source code for the paper:
Texture Generation Using a Graph Generative Adversarial Network and Differentiable Rendering

https://link.springer.com/chapter/10.1007/978-3-031-25825-1_28
    
https://arxiv.org/pdf/2206.08547.pdf


Download ShapeNet car dataset from: https://shapenet.org/

requirements:
    python==3.8
    pytorch==1.9.0+cu111
    pytorch3d==0.6.0

train using [ggan model]:
    sh Experiments/gnn_kraken.sh

test using [ggan model]: 
    sh Experiments/test.sh

Note: The source code contains multiple files used to train other models
Note: The optimization problem is complicated: changing hyperparameters slightly may result in huge changes in the generated texture quality

If you use the source code please cite the following paper:

    @inproceedings{dharma2023texture,
      title={Texture Generation Using a Graph Generative Adversarial Network and Differentiable Rendering},
      author={Dharma, KC and Morrison, Clayton T and Walls, Bradley},
      booktitle={Image and Vision Computing: 37th International Conference, IVCNZ 2022, Auckland, New Zealand, November 24--25, 2022, Revised Selected Papers},
      pages={388--401},
      year={2023},
      organization={Springer}
    }
