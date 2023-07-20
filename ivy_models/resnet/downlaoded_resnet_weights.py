{
    bn1: {
        bias: (<class ivy.data_classes.array.array.Array> shape=[64]),
        running_mean: (<class ivy.data_classes.array.array.Array> shape=[64]),
        running_var: (<class ivy.data_classes.array.array.Array> shape=[64]),
        weight: (<class ivy.data_classes.array.array.Array> shape=[64])
    },
    conv1: {
        weight: (<class ivy.data_classes.array.array.Array> shape=[64, 3, 7, 7])
    },
    fc: {
        bias: (<class ivy.data_classes.array.array.Array> shape=[1000]),
        weight: (<class ivy.data_classes.array.array.Array> shape=[1000, 512])
    },
    layer1: {
        0: {
            bn1: {
                bias: (<class ivy.data_classes.array.array.Array> shape=[64]),
                running_mean: (<class ivy.data_classes.array.array.Array> shape=[64]),
                running_var: (<class ivy.data_classes.array.array.Array> shape=[64]),
                weight: (<class ivy.data_classes.array.array.Array> shape=[64])
            },
            bn2: {
                bias: (<class ivy.data_classes.array.array.Array> shape=[64]),
                running_mean: (<class ivy.data_classes.array.array.Array> shape=[64]),
                running_var: (<class ivy.data_classes.array.array.Array> shape=[64]),
                weight: (<class ivy.data_classes.array.array.Array> shape=[64])
            },
            conv1: {
                weight: (<class ivy.data_classes.array.array.Array> shape=[64, 64, 3, 3])
            },
            conv2: {
                weight: (<class ivy.data_classes.array.array.Array> shape=[64, 64, 3, 3])
            }
        },
        1: {
            bn1: {
                bias: (<class ivy.data_classes.array.array.Array> shape=[64]),
                running_mean: (<class ivy.data_classes.array.array.Array> shape=[64]),
                running_var: (<class ivy.data_classes.array.array.Array> shape=[64]),
                weight: (<class ivy.data_classes.array.array.Array> shape=[64])
            },
            bn2: {
                bias: (<class ivy.data_classes.array.array.Array> shape=[64]),
                running_mean: (<class ivy.data_classes.array.array.Array> shape=[64]),
                running_var: (<class ivy.data_classes.array.array.Array> shape=[64]),
                weight: (<class ivy.data_classes.array.array.Array> shape=[64])
            },
            conv1: {
                weight: (<class ivy.data_classes.array.array.Array> shape=[64, 64, 3, 3])
            },
            conv2: {
                weight: (<class ivy.data_classes.array.array.Array> shape=[64, 64, 3, 3])
            }
        }
    },
    layer2: {
        0: {
            bn1: {
                bias: (<class ivy.data_classes.array.array.Array> shape=[128]),
                running_mean: (<class ivy.data_classes.array.array.Array> shape=[128]),
                running_var: (<class ivy.data_classes.array.array.Array> shape=[128]),
                weight: (<class ivy.data_classes.array.array.Array> shape=[128])
            },
            bn2: {
                bias: (<class ivy.data_classes.array.array.Array> shape=[128]),
                running_mean: (<class ivy.data_classes.array.array.Array> shape=[128]),
                running_var: (<class ivy.data_classes.array.array.Array> shape=[128]),
                weight: (<class ivy.data_classes.array.array.Array> shape=[128])
            },
            conv1: {
                weight: (<class ivy.data_classes.array.array.Array> shape=[128, 64, 3, 3])
            },
            conv2: {
                weight: (<class ivy.data_classes.array.array.Array> shape=[128, 128, 3, 3])
            },
            downsample: {
                0: {
                    weight: (<class ivy.data_classes.array.array.Array> shape=[128, 64, 1, 1])
                },
                1: {
                    bias: (<class ivy.data_classes.array.array.Array> shape=[128]),
                    running_mean: (<class ivy.data_classes.array.array.Array> shape=[128]),
                    running_var: (<class ivy.data_classes.array.array.Array> shape=[128]),
                    weight: (<class ivy.data_classes.array.array.Array> shape=[128])
                }
            }
        },
        1: {
            bn1: {
                bias: (<class ivy.data_classes.array.array.Array> shape=[128]),
                running_mean: (<class ivy.data_classes.array.array.Array> shape=[128]),
                running_var: (<class ivy.data_classes.array.array.Array> shape=[128]),
                weight: (<class ivy.data_classes.array.array.Array> shape=[128])
            },
            bn2: {
                bias: (<class ivy.data_classes.array.array.Array> shape=[128]),
                running_mean: (<class ivy.data_classes.array.array.Array> shape=[128]),
                running_var: (<class ivy.data_classes.array.array.Array> shape=[128]),
                weight: (<class ivy.data_classes.array.array.Array> shape=[128])
            },
            conv1: {
                weight: (<class ivy.data_classes.array.array.Array> shape=[128, 128, 3, 3])
            },
            conv2: {
                weight: (<class ivy.data_classes.array.array.Array> shape=[128, 128, 3, 3])
            }
        }
    },
    layer3: {
        0: {
            bn1: {
                bias: (<class ivy.data_classes.array.array.Array> shape=[256]),
                running_mean: (<class ivy.data_classes.array.array.Array> shape=[256]),
                running_var: (<class ivy.data_classes.array.array.Array> shape=[256]),
                weight: (<class ivy.data_classes.array.array.Array> shape=[256])
            },
            bn2: {
                bias: (<class ivy.data_classes.array.array.Array> shape=[256]),
                running_mean: (<class ivy.data_classes.array.array.Array> shape=[256]),
                running_var: (<class ivy.data_classes.array.array.Array> shape=[256]),
                weight: (<class ivy.data_classes.array.array.Array> shape=[256])
            },
            conv1: {
                weight: (<class ivy.data_classes.array.array.Array> shape=[256, 128, 3, 3])
            },
            conv2: {
                weight: (<class ivy.data_classes.array.array.Array> shape=[256, 256, 3, 3])
            },
            downsample: {
                0: {
                    weight: (<class ivy.data_classes.array.array.Array> shape=[256, 128, 1, 1])
                },
                1: {
                    bias: (<class ivy.data_classes.array.array.Array> shape=[256]),
                    running_mean: (<class ivy.data_classes.array.array.Array> shape=[256]),
                    running_var: (<class ivy.data_classes.array.array.Array> shape=[256]),
                    weight: (<class ivy.data_classes.array.array.Array> shape=[256])
                }
            }
        },
        1: {
            bn1: {
                bias: (<class ivy.data_classes.array.array.Array> shape=[256]),
                running_mean: (<class ivy.data_classes.array.array.Array> shape=[256]),
                running_var: (<class ivy.data_classes.array.array.Array> shape=[256]),
                weight: (<class ivy.data_classes.array.array.Array> shape=[256])
            },
            bn2: {
                bias: (<class ivy.data_classes.array.array.Array> shape=[256]),
                running_mean: (<class ivy.data_classes.array.array.Array> shape=[256]),
                running_var: (<class ivy.data_classes.array.array.Array> shape=[256]),
                weight: (<class ivy.data_classes.array.array.Array> shape=[256])
            },
            conv1: {
                weight: (<class ivy.data_classes.array.array.Array> shape=[256, 256, 3, 3])
            },
            conv2: {
                weight: (<class ivy.data_classes.array.array.Array> shape=[256, 256, 3, 3])
            }
        }
    },
    layer4: {
        0: {
            bn1: {
                bias: (<class ivy.data_classes.array.array.Array> shape=[512]),
                running_mean: (<class ivy.data_classes.array.array.Array> shape=[512]),
                running_var: (<class ivy.data_classes.array.array.Array> shape=[512]),
                weight: (<class ivy.data_classes.array.array.Array> shape=[512])
            },
            bn2: {
                bias: (<class ivy.data_classes.array.array.Array> shape=[512]),
                running_mean: (<class ivy.data_classes.array.array.Array> shape=[512]),
                running_var: (<class ivy.data_classes.array.array.Array> shape=[512]),
                weight: (<class ivy.data_classes.array.array.Array> shape=[512])
            },
            conv1: {
                weight: (<class ivy.data_classes.array.array.Array> shape=[512, 256, 3, 3])
            },
            conv2: {
                weight: (<class ivy.data_classes.array.array.Array> shape=[512, 512, 3, 3])
            },
            downsample: {
                0: {
                    weight: (<class ivy.data_classes.array.array.Array> shape=[512, 256, 1, 1])
                },
                1: {
                    bias: (<class ivy.data_classes.array.array.Array> shape=[512]),
                    running_mean: (<class ivy.data_classes.array.array.Array> shape=[512]),
                    running_var: (<class ivy.data_classes.array.array.Array> shape=[512]),
                    weight: (<class ivy.data_classes.array.array.Array> shape=[512])
                }
            }
        },
        1: {
            bn1: {
                bias: (<class ivy.data_classes.array.array.Array> shape=[512]),
                running_mean: (<class ivy.data_classes.array.array.Array> shape=[512]),
                running_var: (<class ivy.data_classes.array.array.Array> shape=[512]),
                weight: (<class ivy.data_classes.array.array.Array> shape=[512])
            },
            bn2: {
                bias: (<class ivy.data_classes.array.array.Array> shape=[512]),
                running_mean: (<class ivy.data_classes.array.array.Array> shape=[512]),
                running_var: (<class ivy.data_classes.array.array.Array> shape=[512]),
                weight: (<class ivy.data_classes.array.array.Array> shape=[512])
            },
            conv1: {
                weight: (<class ivy.data_classes.array.array.Array> shape=[512, 512, 3, 3])
            },
            conv2: {
                weight: (<class ivy.data_classes.array.array.Array> shape=[512, 512, 3, 3])
            }
        }
    }
}