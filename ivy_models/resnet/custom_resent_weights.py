{
    bn1: {
        b: (<class ivy.data_classes.array.array.Array> shape=[64]),
        running_mean: (<class ivy.data_classes.array.array.Array> shape=[64]),
        running_var: (<class ivy.data_classes.array.array.Array> shape=[64]),
        w: (<class ivy.data_classes.array.array.Array> shape=[64])
    },
    conv1: {
        w: (<class ivy.data_classes.array.array.Array> shape=[7, 7, 3, 64])
    },
    fc: {
        b: (<class ivy.data_classes.array.array.Array> shape=[1000]),
        w: (<class ivy.data_classes.array.array.Array> shape=[1000, 512])
    },
    layer1: {
        submodules: {
            v0: {
                bn1: {
                    b: (<class ivy.data_classes.array.array.Array> shape=[64]),
                    running_mean: (<class ivy.data_classes.array.array.Array> shape=[64]),
                    running_var: (<class ivy.data_classes.array.array.Array> shape=[64]),
                    w: (<class ivy.data_classes.array.array.Array> shape=[64])
                },
                bn2: {
                    b: (<class ivy.data_classes.array.array.Array> shape=[64]),
                    running_mean: (<class ivy.data_classes.array.array.Array> shape=[64]),
                    running_var: (<class ivy.data_classes.array.array.Array> shape=[64]),
                    w: (<class ivy.data_classes.array.array.Array> shape=[64])
                },
                conv1: {
                    w: (<class ivy.data_classes.array.array.Array> shape=[3, 3, 64, 64])
                },
                conv2: {
                    w: (<class ivy.data_classes.array.array.Array> shape=[3, 3, 64, 64])
                }
            },
            v1: {
                bn1: {
                    b: (<class ivy.data_classes.array.array.Array> shape=[64]),
                    running_mean: (<class ivy.data_classes.array.array.Array> shape=[64]),
                    running_var: (<class ivy.data_classes.array.array.Array> shape=[64]),
                    w: (<class ivy.data_classes.array.array.Array> shape=[64])
                },
                bn2: {
                    b: (<class ivy.data_classes.array.array.Array> shape=[64]),
                    running_mean: (<class ivy.data_classes.array.array.Array> shape=[64]),
                    running_var: (<class ivy.data_classes.array.array.Array> shape=[64]),
                    w: (<class ivy.data_classes.array.array.Array> shape=[64])
                },
                conv1: {
                    w: (<class ivy.data_classes.array.array.Array> shape=[3, 3, 64, 64])
                },
                conv2: {
                    w: (<class ivy.data_classes.array.array.Array> shape=[3, 3, 64, 64])
                }
            }
        }
    },
    layer2: {
        submodules: {
            v0: {
                bn1: {
                    b: (<class ivy.data_classes.array.array.Array> shape=[128]),
                    running_mean: (<class ivy.data_classes.array.array.Array> shape=[128]),
                    running_var: (<class ivy.data_classes.array.array.Array> shape=[128]),
                    w: (<class ivy.data_classes.array.array.Array> shape=[128])
                },
                bn2: {
                    b: (<class ivy.data_classes.array.array.Array> shape=[128]),
                    running_mean: (<class ivy.data_classes.array.array.Array> shape=[128]),
                    running_var: (<class ivy.data_classes.array.array.Array> shape=[128]),
                    w: (<class ivy.data_classes.array.array.Array> shape=[128])
                },
                conv1: {
                    w: (<class ivy.data_classes.array.array.Array> shape=[3, 3, 64, 128])
                },
                conv2: {
                    w: (<class ivy.data_classes.array.array.Array> shape=[3, 3, 128, 128])
                },
                downsample: {
                    submodules: {
                        v0: {
                            w: (<class ivy.data_classes.array.array.Array> shape=[1, 1, 64, 128])
                        },
                        v1: {
                            b: (<class ivy.data_classes.array.array.Array> shape=[128]),
                            running_mean: (<class ivy.data_classes.array.array.Array> shape=[128]),
                            running_var: (<class ivy.data_classes.array.array.Array> shape=[128]),
                            w: (<class ivy.data_classes.array.array.Array> shape=[128])
                        }
                    }
                }
            },
            v1: {
                bn1: {
                    b: (<class ivy.data_classes.array.array.Array> shape=[128]),
                    running_mean: (<class ivy.data_classes.array.array.Array> shape=[128]),
                    running_var: (<class ivy.data_classes.array.array.Array> shape=[128]),
                    w: (<class ivy.data_classes.array.array.Array> shape=[128])
                },
                bn2: {
                    b: (<class ivy.data_classes.array.array.Array> shape=[128]),
                    running_mean: (<class ivy.data_classes.array.array.Array> shape=[128]),
                    running_var: (<class ivy.data_classes.array.array.Array> shape=[128]),
                    w: (<class ivy.data_classes.array.array.Array> shape=[128])
                },
                conv1: {
                    w: (<class ivy.data_classes.array.array.Array> shape=[3, 3, 128, 128])
                },
                conv2: {
                    w: (<class ivy.data_classes.array.array.Array> shape=[3, 3, 128, 128])
                }
            }
        }
    },
    layer3: {
        submodules: {
            v0: {
                bn1: {
                    b: (<class ivy.data_classes.array.array.Array> shape=[256]),
                    running_mean: (<class ivy.data_classes.array.array.Array> shape=[256]),
                    running_var: (<class ivy.data_classes.array.array.Array> shape=[256]),
                    w: (<class ivy.data_classes.array.array.Array> shape=[256])
                },
                bn2: {
                    b: (<class ivy.data_classes.array.array.Array> shape=[256]),
                    running_mean: (<class ivy.data_classes.array.array.Array> shape=[256]),
                    running_var: (<class ivy.data_classes.array.array.Array> shape=[256]),
                    w: (<class ivy.data_classes.array.array.Array> shape=[256])
                },
                conv1: {
                    w: (<class ivy.data_classes.array.array.Array> shape=[3, 3, 128, 256])
                },
                conv2: {
                    w: (<class ivy.data_classes.array.array.Array> shape=[3, 3, 256, 256])
                },
                downsample: {
                    submodules: {
                        v0: {
                            w: (<class ivy.data_classes.array.array.Array> shape=[1, 1, 128, 256])
                        },
                        v1: {
                            b: (<class ivy.data_classes.array.array.Array> shape=[256]),
                            running_mean: (<class ivy.data_classes.array.array.Array> shape=[256]),
                            running_var: (<class ivy.data_classes.array.array.Array> shape=[256]),
                            w: (<class ivy.data_classes.array.array.Array> shape=[256])
                        }
                    }
                }
            },
            v1: {
                bn1: {
                    b: (<class ivy.data_classes.array.array.Array> shape=[256]),
                    running_mean: (<class ivy.data_classes.array.array.Array> shape=[256]),
                    running_var: (<class ivy.data_classes.array.array.Array> shape=[256]),
                    w: (<class ivy.data_classes.array.array.Array> shape=[256])
                },
                bn2: {
                    b: (<class ivy.data_classes.array.array.Array> shape=[256]),
                    running_mean: (<class ivy.data_classes.array.array.Array> shape=[256]),
                    running_var: (<class ivy.data_classes.array.array.Array> shape=[256]),
                    w: (<class ivy.data_classes.array.array.Array> shape=[256])
                },
                conv1: {
                    w: (<class ivy.data_classes.array.array.Array> shape=[3, 3, 256, 256])
                },
                conv2: {
                    w: (<class ivy.data_classes.array.array.Array> shape=[3, 3, 256, 256])
                }
            }
        }
    },
    layer4: {
        submodules: {
            v0: {
                bn1: {
                    b: (<class ivy.data_classes.array.array.Array> shape=[512]),
                    running_mean: (<class ivy.data_classes.array.array.Array> shape=[512]),
                    running_var: (<class ivy.data_classes.array.array.Array> shape=[512]),
                    w: (<class ivy.data_classes.array.array.Array> shape=[512])
                },
                bn2: {
                    b: (<class ivy.data_classes.array.array.Array> shape=[512]),
                    running_mean: (<class ivy.data_classes.array.array.Array> shape=[512]),
                    running_var: (<class ivy.data_classes.array.array.Array> shape=[512]),
                    w: (<class ivy.data_classes.array.array.Array> shape=[512])
                },
                conv1: {
                    w: (<class ivy.data_classes.array.array.Array> shape=[3, 3, 256, 512])
                },
                conv2: {
                    w: (<class ivy.data_classes.array.array.Array> shape=[3, 3, 512, 512])
                },
                downsample: {
                    submodules: {
                        v0: {
                            w: (<class ivy.data_classes.array.array.Array> shape=[1, 1, 256, 512])
                        },
                        v1: {
                            b: (<class ivy.data_classes.array.array.Array> shape=[512]),
                            running_mean: (<class ivy.data_classes.array.array.Array> shape=[512]),
                            running_var: (<class ivy.data_classes.array.array.Array> shape=[512]),
                            w: (<class ivy.data_classes.array.array.Array> shape=[512])
                        }
                    }
                }
            },
            v1: {
                bn1: {
                    b: (<class ivy.data_classes.array.array.Array> shape=[512]),
                    running_mean: (<class ivy.data_classes.array.array.Array> shape=[512]),
                    running_var: (<class ivy.data_classes.array.array.Array> shape=[512]),
                    w: (<class ivy.data_classes.array.array.Array> shape=[512])
                },
                bn2: {
                    b: (<class ivy.data_classes.array.array.Array> shape=[512]),
                    running_mean: (<class ivy.data_classes.array.array.Array> shape=[512]),
                    running_var: (<class ivy.data_classes.array.array.Array> shape=[512]),
                    w: (<class ivy.data_classes.array.array.Array> shape=[512])
                },
                conv1: {
                    w: (<class ivy.data_classes.array.array.Array> shape=[3, 3, 512, 512])
                },
                conv2: {
                    w: (<class ivy.data_classes.array.array.Array> shape=[3, 3, 512, 512])
                }
            }
        }
    }
}