{
    aux1: {
        conv: {
            bn: {
                bias: (<class ivy.data_classes.array.array.Array> shape=[128]),
                num_batches_tracked: ivy.array(0),
                running_mean: (<class ivy.data_classes.array.array.Array> shape=[128]),
                running_var: (<class ivy.data_classes.array.array.Array> shape=[128]),
                weight: (<class ivy.data_classes.array.array.Array> shape=[128])
            },
            conv: {
                weight: (<class ivy.data_classes.array.array.Array> shape=[128, 512, 1, 1])
            }
        },
        fc1: {
            bias: (<class ivy.data_classes.array.array.Array> shape=[1024]),
            weight: (<class ivy.data_classes.array.array.Array> shape=[1024, 2048])
        },
        fc2: {
            bias: (<class ivy.data_classes.array.array.Array> shape=[1000]),
            weight: (<class ivy.data_classes.array.array.Array> shape=[1000, 1024])
        }
    },
    aux2: {
        conv: {
            bn: {
                bias: (<class ivy.data_classes.array.array.Array> shape=[128]),
                num_batches_tracked: ivy.array(0),
                running_mean: (<class ivy.data_classes.array.array.Array> shape=[128]),
                running_var: (<class ivy.data_classes.array.array.Array> shape=[128]),
                weight: (<class ivy.data_classes.array.array.Array> shape=[128])
            },
            conv: {
                weight: (<class ivy.data_classes.array.array.Array> shape=[128, 528, 1, 1])
            }
        },
        fc1: {
            bias: (<class ivy.data_classes.array.array.Array> shape=[1024]),
            weight: (<class ivy.data_classes.array.array.Array> shape=[1024, 2048])
        },
        fc2: {
            bias: (<class ivy.data_classes.array.array.Array> shape=[1000]),
            weight: (<class ivy.data_classes.array.array.Array> shape=[1000, 1024])
        }
    },
    conv1: {
        bn: {
            bias: (<class ivy.data_classes.array.array.Array> shape=[64]),
            num_batches_tracked: ivy.array(0),
            running_mean: (<class ivy.data_classes.array.array.Array> shape=[64]),
            running_var: (<class ivy.data_classes.array.array.Array> shape=[64]),
            weight: (<class ivy.data_classes.array.array.Array> shape=[64])
        },
        conv: {
            weight: (<class ivy.data_classes.array.array.Array> shape=[64, 3, 7, 7])
        }
    },
    conv2: {
        bn: {
            bias: (<class ivy.data_classes.array.array.Array> shape=[64]),
            num_batches_tracked: ivy.array(0),
            running_mean: (<class ivy.data_classes.array.array.Array> shape=[64]),
            running_var: (<class ivy.data_classes.array.array.Array> shape=[64]),
            weight: (<class ivy.data_classes.array.array.Array> shape=[64])
        },
        conv: {
            weight: (<class ivy.data_classes.array.array.Array> shape=[64, 64, 1, 1])
        }
    },
    conv3: {
        bn: {
            bias: (<class ivy.data_classes.array.array.Array> shape=[192]),
            num_batches_tracked: ivy.array(0),
            running_mean: (<class ivy.data_classes.array.array.Array> shape=[192]),
            running_var: (<class ivy.data_classes.array.array.Array> shape=[192]),
            weight: (<class ivy.data_classes.array.array.Array> shape=[192])
        },
        conv: {
            weight: (<class ivy.data_classes.array.array.Array> shape=[192, 64, 3, 3])
        }
    },
    fc: {
        bias: (<class ivy.data_classes.array.array.Array> shape=[1000]),
        weight: (<class ivy.data_classes.array.array.Array> shape=[1000, 1024])
    },
    inception3a: {
        branch1: {
            bn: {
                bias: (<class ivy.data_classes.array.array.Array> shape=[64]),
                num_batches_tracked: ivy.array(0),
                running_mean: (<class ivy.data_classes.array.array.Array> shape=[64]),
                running_var: (<class ivy.data_classes.array.array.Array> shape=[64]),
                weight: (<class ivy.data_classes.array.array.Array> shape=[64])
            },
            conv: {
                weight: (<class ivy.data_classes.array.array.Array> shape=[64, 192, 1, 1])
            }
        },
        branch2: {
            0: {
                bn: {
                    bias: (<class ivy.data_classes.array.array.Array> shape=[96]),
                    num_batches_tracked: ivy.array(0),
                    running_mean: (<class ivy.data_classes.array.array.Array> shape=[96]),
                    running_var: (<class ivy.data_classes.array.array.Array> shape=[96]),
                    weight: (<class ivy.data_classes.array.array.Array> shape=[96])
                },
                conv: {
                    weight: (<class ivy.data_classes.array.array.Array> shape=[96, 192, 1, 1])
                }
            },
            1: {
                bn: {
                    bias: (<class ivy.data_classes.array.array.Array> shape=[128]),
                    num_batches_tracked: ivy.array(0),
                    running_mean: (<class ivy.data_classes.array.array.Array> shape=[128]),
                    running_var: (<class ivy.data_classes.array.array.Array> shape=[128]),
                    weight: (<class ivy.data_classes.array.array.Array> shape=[128])
                },
                conv: {
                    weight: (<class ivy.data_classes.array.array.Array> shape=[128, 96, 3, 3])
                }
            }
        },
        branch3: {
            0: {
                bn: {
                    bias: (<class ivy.data_classes.array.array.Array> shape=[16]),
                    num_batches_tracked: ivy.array(0),
                    running_mean: (<class ivy.data_classes.array.array.Array> shape=[16]),
                    running_var: (<class ivy.data_classes.array.array.Array> shape=[16]),
                    weight: (<class ivy.data_classes.array.array.Array> shape=[16])
                },
                conv: {
                    weight: (<class ivy.data_classes.array.array.Array> shape=[16, 192, 1, 1])
                }
            },
            1: {
                bn: {
                    bias: (<class ivy.data_classes.array.array.Array> shape=[32]),
                    num_batches_tracked: ivy.array(0),
                    running_mean: (<class ivy.data_classes.array.array.Array> shape=[32]),
                    running_var: (<class ivy.data_classes.array.array.Array> shape=[32]),
                    weight: (<class ivy.data_classes.array.array.Array> shape=[32])
                },
                conv: {
                    weight: (<class ivy.data_classes.array.array.Array> shape=[32, 16, 3, 3])
                }
            }
        },
        branch4: {
            1: {
                bn: {
                    bias: (<class ivy.data_classes.array.array.Array> shape=[32]),
                    num_batches_tracked: ivy.array(0),
                    running_mean: (<class ivy.data_classes.array.array.Array> shape=[32]),
                    running_var: (<class ivy.data_classes.array.array.Array> shape=[32]),
                    weight: (<class ivy.data_classes.array.array.Array> shape=[32])
                },
                conv: {
                    weight: (<class ivy.data_classes.array.array.Array> shape=[32, 192, 1, 1])
                }
            }
        }
    },
    inception3b: {
        branch1: {
            bn: {
                bias: (<class ivy.data_classes.array.array.Array> shape=[128]),
                num_batches_tracked: ivy.array(0),
                running_mean: (<class ivy.data_classes.array.array.Array> shape=[128]),
                running_var: (<class ivy.data_classes.array.array.Array> shape=[128]),
                weight: (<class ivy.data_classes.array.array.Array> shape=[128])
            },
            conv: {
                weight: (<class ivy.data_classes.array.array.Array> shape=[128, 256, 1, 1])
            }
        },
        branch2: {
            0: {
                bn: {
                    bias: (<class ivy.data_classes.array.array.Array> shape=[128]),
                    num_batches_tracked: ivy.array(0),
                    running_mean: (<class ivy.data_classes.array.array.Array> shape=[128]),
                    running_var: (<class ivy.data_classes.array.array.Array> shape=[128]),
                    weight: (<class ivy.data_classes.array.array.Array> shape=[128])
                },
                conv: {
                    weight: (<class ivy.data_classes.array.array.Array> shape=[128, 256, 1, 1])
                }
            },
            1: {
                bn: {
                    bias: (<class ivy.data_classes.array.array.Array> shape=[192]),
                    num_batches_tracked: ivy.array(0),
                    running_mean: (<class ivy.data_classes.array.array.Array> shape=[192]),
                    running_var: (<class ivy.data_classes.array.array.Array> shape=[192]),
                    weight: (<class ivy.data_classes.array.array.Array> shape=[192])
                },
                conv: {
                    weight: (<class ivy.data_classes.array.array.Array> shape=[192, 128, 3, 3])
                }
            }
        },
        branch3: {
            0: {
                bn: {
                    bias: (<class ivy.data_classes.array.array.Array> shape=[32]),
                    num_batches_tracked: ivy.array(0),
                    running_mean: (<class ivy.data_classes.array.array.Array> shape=[32]),
                    running_var: (<class ivy.data_classes.array.array.Array> shape=[32]),
                    weight: (<class ivy.data_classes.array.array.Array> shape=[32])
                },
                conv: {
                    weight: (<class ivy.data_classes.array.array.Array> shape=[32, 256, 1, 1])
                }
            },
            1: {
                bn: {
                    bias: (<class ivy.data_classes.array.array.Array> shape=[96]),
                    num_batches_tracked: ivy.array(0),
                    running_mean: (<class ivy.data_classes.array.array.Array> shape=[96]),
                    running_var: (<class ivy.data_classes.array.array.Array> shape=[96]),
                    weight: (<class ivy.data_classes.array.array.Array> shape=[96])
                },
                conv: {
                    weight: (<class ivy.data_classes.array.array.Array> shape=[96, 32, 3, 3])
                }
            }
        },
        branch4: {
            1: {
                bn: {
                    bias: (<class ivy.data_classes.array.array.Array> shape=[64]),
                    num_batches_tracked: ivy.array(0),
                    running_mean: (<class ivy.data_classes.array.array.Array> shape=[64]),
                    running_var: (<class ivy.data_classes.array.array.Array> shape=[64]),
                    weight: (<class ivy.data_classes.array.array.Array> shape=[64])
                },
                conv: {
                    weight: (<class ivy.data_classes.array.array.Array> shape=[64, 256, 1, 1])
                }
            }
        }
    },
    inception4a: {
        branch1: {
            bn: {
                bias: (<class ivy.data_classes.array.array.Array> shape=[192]),
                num_batches_tracked: ivy.array(0),
                running_mean: (<class ivy.data_classes.array.array.Array> shape=[192]),
                running_var: (<class ivy.data_classes.array.array.Array> shape=[192]),
                weight: (<class ivy.data_classes.array.array.Array> shape=[192])
            },
            conv: {
                weight: (<class ivy.data_classes.array.array.Array> shape=[192, 480, 1, 1])
            }
        },
        branch2: {
            0: {
                bn: {
                    bias: (<class ivy.data_classes.array.array.Array> shape=[96]),
                    num_batches_tracked: ivy.array(0),
                    running_mean: (<class ivy.data_classes.array.array.Array> shape=[96]),
                    running_var: (<class ivy.data_classes.array.array.Array> shape=[96]),
                    weight: (<class ivy.data_classes.array.array.Array> shape=[96])
                },
                conv: {
                    weight: (<class ivy.data_classes.array.array.Array> shape=[96, 480, 1, 1])
                }
            },
            1: {
                bn: {
                    bias: (<class ivy.data_classes.array.array.Array> shape=[208]),
                    num_batches_tracked: ivy.array(0),
                    running_mean: (<class ivy.data_classes.array.array.Array> shape=[208]),
                    running_var: (<class ivy.data_classes.array.array.Array> shape=[208]),
                    weight: (<class ivy.data_classes.array.array.Array> shape=[208])
                },
                conv: {
                    weight: (<class ivy.data_classes.array.array.Array> shape=[208, 96, 3, 3])
                }
            }
        },
        branch3: {
            0: {
                bn: {
                    bias: (<class ivy.data_classes.array.array.Array> shape=[16]),
                    num_batches_tracked: ivy.array(0),
                    running_mean: (<class ivy.data_classes.array.array.Array> shape=[16]),
                    running_var: (<class ivy.data_classes.array.array.Array> shape=[16]),
                    weight: (<class ivy.data_classes.array.array.Array> shape=[16])
                },
                conv: {
                    weight: (<class ivy.data_classes.array.array.Array> shape=[16, 480, 1, 1])
                }
            },
            1: {
                bn: {
                    bias: (<class ivy.data_classes.array.array.Array> shape=[48]),
                    num_batches_tracked: ivy.array(0),
                    running_mean: (<class ivy.data_classes.array.array.Array> shape=[48]),
                    running_var: (<class ivy.data_classes.array.array.Array> shape=[48]),
                    weight: (<class ivy.data_classes.array.array.Array> shape=[48])
                },
                conv: {
                    weight: (<class ivy.data_classes.array.array.Array> shape=[48, 16, 3, 3])
                }
            }
        },
        branch4: {
            1: {
                bn: {
                    bias: (<class ivy.data_classes.array.array.Array> shape=[64]),
                    num_batches_tracked: ivy.array(0),
                    running_mean: (<class ivy.data_classes.array.array.Array> shape=[64]),
                    running_var: (<class ivy.data_classes.array.array.Array> shape=[64]),
                    weight: (<class ivy.data_classes.array.array.Array> shape=[64])
                },
                conv: {
                    weight: (<class ivy.data_classes.array.array.Array> shape=[64, 480, 1, 1])
                }
            }
        }
    },
    inception4b: {
        branch1: {
            bn: {
                bias: (<class ivy.data_classes.array.array.Array> shape=[160]),
                num_batches_tracked: ivy.array(0),
                running_mean: (<class ivy.data_classes.array.array.Array> shape=[160]),
                running_var: (<class ivy.data_classes.array.array.Array> shape=[160]),
                weight: (<class ivy.data_classes.array.array.Array> shape=[160])
            },
            conv: {
                weight: (<class ivy.data_classes.array.array.Array> shape=[160, 512, 1, 1])
            }
        },
        branch2: {
            0: {
                bn: {
                    bias: (<class ivy.data_classes.array.array.Array> shape=[112]),
                    num_batches_tracked: ivy.array(0),
                    running_mean: (<class ivy.data_classes.array.array.Array> shape=[112]),
                    running_var: (<class ivy.data_classes.array.array.Array> shape=[112]),
                    weight: (<class ivy.data_classes.array.array.Array> shape=[112])
                },
                conv: {
                    weight: (<class ivy.data_classes.array.array.Array> shape=[112, 512, 1, 1])
                }
            },
            1: {
                bn: {
                    bias: (<class ivy.data_classes.array.array.Array> shape=[224]),
                    num_batches_tracked: ivy.array(0),
                    running_mean: (<class ivy.data_classes.array.array.Array> shape=[224]),
                    running_var: (<class ivy.data_classes.array.array.Array> shape=[224]),
                    weight: (<class ivy.data_classes.array.array.Array> shape=[224])
                },
                conv: {
                    weight: (<class ivy.data_classes.array.array.Array> shape=[224, 112, 3, 3])
                }
            }
        },
        branch3: {
            0: {
                bn: {
                    bias: (<class ivy.data_classes.array.array.Array> shape=[24]),
                    num_batches_tracked: ivy.array(0),
                    running_mean: (<class ivy.data_classes.array.array.Array> shape=[24]),
                    running_var: (<class ivy.data_classes.array.array.Array> shape=[24]),
                    weight: (<class ivy.data_classes.array.array.Array> shape=[24])
                },
                conv: {
                    weight: (<class ivy.data_classes.array.array.Array> shape=[24, 512, 1, 1])
                }
            },
            1: {
                bn: {
                    bias: (<class ivy.data_classes.array.array.Array> shape=[64]),
                    num_batches_tracked: ivy.array(0),
                    running_mean: (<class ivy.data_classes.array.array.Array> shape=[64]),
                    running_var: (<class ivy.data_classes.array.array.Array> shape=[64]),
                    weight: (<class ivy.data_classes.array.array.Array> shape=[64])
                },
                conv: {
                    weight: (<class ivy.data_classes.array.array.Array> shape=[64, 24, 3, 3])
                }
            }
        },
        branch4: {
            1: {
                bn: {
                    bias: (<class ivy.data_classes.array.array.Array> shape=[64]),
                    num_batches_tracked: ivy.array(0),
                    running_mean: (<class ivy.data_classes.array.array.Array> shape=[64]),
                    running_var: (<class ivy.data_classes.array.array.Array> shape=[64]),
                    weight: (<class ivy.data_classes.array.array.Array> shape=[64])
                },
                conv: {
                    weight: (<class ivy.data_classes.array.array.Array> shape=[64, 512, 1, 1])
                }
            }
        }
    },
    inception4c: {
        branch1: {
            bn: {
                bias: (<class ivy.data_classes.array.array.Array> shape=[128]),
                num_batches_tracked: ivy.array(0),
                running_mean: (<class ivy.data_classes.array.array.Array> shape=[128]),
                running_var: (<class ivy.data_classes.array.array.Array> shape=[128]),
                weight: (<class ivy.data_classes.array.array.Array> shape=[128])
            },
            conv: {
                weight: (<class ivy.data_classes.array.array.Array> shape=[128, 512, 1, 1])
            }
        },
        branch2: {
            0: {
                bn: {
                    bias: (<class ivy.data_classes.array.array.Array> shape=[128]),
                    num_batches_tracked: ivy.array(0),
                    running_mean: (<class ivy.data_classes.array.array.Array> shape=[128]),
                    running_var: (<class ivy.data_classes.array.array.Array> shape=[128]),
                    weight: (<class ivy.data_classes.array.array.Array> shape=[128])
                },
                conv: {
                    weight: (<class ivy.data_classes.array.array.Array> shape=[128, 512, 1, 1])
                }
            },
            1: {
                bn: {
                    bias: (<class ivy.data_classes.array.array.Array> shape=[256]),
                    num_batches_tracked: ivy.array(0),
                    running_mean: (<class ivy.data_classes.array.array.Array> shape=[256]),
                    running_var: (<class ivy.data_classes.array.array.Array> shape=[256]),
                    weight: (<class ivy.data_classes.array.array.Array> shape=[256])
                },
                conv: {
                    weight: (<class ivy.data_classes.array.array.Array> shape=[256, 128, 3, 3])
                }
            }
        },
        branch3: {
            0: {
                bn: {
                    bias: (<class ivy.data_classes.array.array.Array> shape=[24]),
                    num_batches_tracked: ivy.array(0),
                    running_mean: (<class ivy.data_classes.array.array.Array> shape=[24]),
                    running_var: (<class ivy.data_classes.array.array.Array> shape=[24]),
                    weight: (<class ivy.data_classes.array.array.Array> shape=[24])
                },
                conv: {
                    weight: (<class ivy.data_classes.array.array.Array> shape=[24, 512, 1, 1])
                }
            },
            1: {
                bn: {
                    bias: (<class ivy.data_classes.array.array.Array> shape=[64]),
                    num_batches_tracked: ivy.array(0),
                    running_mean: (<class ivy.data_classes.array.array.Array> shape=[64]),
                    running_var: (<class ivy.data_classes.array.array.Array> shape=[64]),
                    weight: (<class ivy.data_classes.array.array.Array> shape=[64])
                },
                conv: {
                    weight: (<class ivy.data_classes.array.array.Array> shape=[64, 24, 3, 3])
                }
            }
        },
        branch4: {
            1: {
                bn: {
                    bias: (<class ivy.data_classes.array.array.Array> shape=[64]),
                    num_batches_tracked: ivy.array(0),
                    running_mean: (<class ivy.data_classes.array.array.Array> shape=[64]),
                    running_var: (<class ivy.data_classes.array.array.Array> shape=[64]),
                    weight: (<class ivy.data_classes.array.array.Array> shape=[64])
                },
                conv: {
                    weight: (<class ivy.data_classes.array.array.Array> shape=[64, 512, 1, 1])
                }
            }
        }
    },
    inception4d: {
        branch1: {
            bn: {
                bias: (<class ivy.data_classes.array.array.Array> shape=[112]),
                num_batches_tracked: ivy.array(0),
                running_mean: (<class ivy.data_classes.array.array.Array> shape=[112]),
                running_var: (<class ivy.data_classes.array.array.Array> shape=[112]),
                weight: (<class ivy.data_classes.array.array.Array> shape=[112])
            },
            conv: {
                weight: (<class ivy.data_classes.array.array.Array> shape=[112, 512, 1, 1])
            }
        },
        branch2: {
            0: {
                bn: {
                    bias: (<class ivy.data_classes.array.array.Array> shape=[144]),
                    num_batches_tracked: ivy.array(0),
                    running_mean: (<class ivy.data_classes.array.array.Array> shape=[144]),
                    running_var: (<class ivy.data_classes.array.array.Array> shape=[144]),
                    weight: (<class ivy.data_classes.array.array.Array> shape=[144])
                },
                conv: {
                    weight: (<class ivy.data_classes.array.array.Array> shape=[144, 512, 1, 1])
                }
            },
            1: {
                bn: {
                    bias: (<class ivy.data_classes.array.array.Array> shape=[288]),
                    num_batches_tracked: ivy.array(0),
                    running_mean: (<class ivy.data_classes.array.array.Array> shape=[288]),
                    running_var: (<class ivy.data_classes.array.array.Array> shape=[288]),
                    weight: (<class ivy.data_classes.array.array.Array> shape=[288])
                },
                conv: {
                    weight: (<class ivy.data_classes.array.array.Array> shape=[288, 144, 3, 3])
                }
            }
        },
        branch3: {
            0: {
                bn: {
                    bias: (<class ivy.data_classes.array.array.Array> shape=[32]),
                    num_batches_tracked: ivy.array(0),
                    running_mean: (<class ivy.data_classes.array.array.Array> shape=[32]),
                    running_var: (<class ivy.data_classes.array.array.Array> shape=[32]),
                    weight: (<class ivy.data_classes.array.array.Array> shape=[32])
                },
                conv: {
                    weight: (<class ivy.data_classes.array.array.Array> shape=[32, 512, 1, 1])
                }
            },
            1: {
                bn: {
                    bias: (<class ivy.data_classes.array.array.Array> shape=[64]),
                    num_batches_tracked: ivy.array(0),
                    running_mean: (<class ivy.data_classes.array.array.Array> shape=[64]),
                    running_var: (<class ivy.data_classes.array.array.Array> shape=[64]),
                    weight: (<class ivy.data_classes.array.array.Array> shape=[64])
                },
                conv: {
                    weight: (<class ivy.data_classes.array.array.Array> shape=[64, 32, 3, 3])
                }
            }
        },
        branch4: {
            1: {
                bn: {
                    bias: (<class ivy.data_classes.array.array.Array> shape=[64]),
                    num_batches_tracked: ivy.array(0),
                    running_mean: (<class ivy.data_classes.array.array.Array> shape=[64]),
                    running_var: (<class ivy.data_classes.array.array.Array> shape=[64]),
                    weight: (<class ivy.data_classes.array.array.Array> shape=[64])
                },
                conv: {
                    weight: (<class ivy.data_classes.array.array.Array> shape=[64, 512, 1, 1])
                }
            }
        }
    },
    inception4e: {
        branch1: {
            bn: {
                bias: (<class ivy.data_classes.array.array.Array> shape=[256]),
                num_batches_tracked: ivy.array(0),
                running_mean: (<class ivy.data_classes.array.array.Array> shape=[256]),
                running_var: (<class ivy.data_classes.array.array.Array> shape=[256]),
                weight: (<class ivy.data_classes.array.array.Array> shape=[256])
            },
            conv: {
                weight: (<class ivy.data_classes.array.array.Array> shape=[256, 528, 1, 1])
            }
        },
        branch2: {
            0: {
                bn: {
                    bias: (<class ivy.data_classes.array.array.Array> shape=[160]),
                    num_batches_tracked: ivy.array(0),
                    running_mean: (<class ivy.data_classes.array.array.Array> shape=[160]),
                    running_var: (<class ivy.data_classes.array.array.Array> shape=[160]),
                    weight: (<class ivy.data_classes.array.array.Array> shape=[160])
                },
                conv: {
                    weight: (<class ivy.data_classes.array.array.Array> shape=[160, 528, 1, 1])
                }
            },
            1: {
                bn: {
                    bias: (<class ivy.data_classes.array.array.Array> shape=[320]),
                    num_batches_tracked: ivy.array(0),
                    running_mean: (<class ivy.data_classes.array.array.Array> shape=[320]),
                    running_var: (<class ivy.data_classes.array.array.Array> shape=[320]),
                    weight: (<class ivy.data_classes.array.array.Array> shape=[320])
                },
                conv: {
                    weight: (<class ivy.data_classes.array.array.Array> shape=[320, 160, 3, 3])
                }
            }
        },
        branch3: {
            0: {
                bn: {
                    bias: (<class ivy.data_classes.array.array.Array> shape=[32]),
                    num_batches_tracked: ivy.array(0),
                    running_mean: (<class ivy.data_classes.array.array.Array> shape=[32]),
                    running_var: (<class ivy.data_classes.array.array.Array> shape=[32]),
                    weight: (<class ivy.data_classes.array.array.Array> shape=[32])
                },
                conv: {
                    weight: (<class ivy.data_classes.array.array.Array> shape=[32, 528, 1, 1])
                }
            },
            1: {
                bn: {
                    bias: (<class ivy.data_classes.array.array.Array> shape=[128]),
                    num_batches_tracked: ivy.array(0),
                    running_mean: (<class ivy.data_classes.array.array.Array> shape=[128]),
                    running_var: (<class ivy.data_classes.array.array.Array> shape=[128]),
                    weight: (<class ivy.data_classes.array.array.Array> shape=[128])
                },
                conv: {
                    weight: (<class ivy.data_classes.array.array.Array> shape=[128, 32, 3, 3])
                }
            }
        },
        branch4: {
            1: {
                bn: {
                    bias: (<class ivy.data_classes.array.array.Array> shape=[128]),
                    num_batches_tracked: ivy.array(0),
                    running_mean: (<class ivy.data_classes.array.array.Array> shape=[128]),
                    running_var: (<class ivy.data_classes.array.array.Array> shape=[128]),
                    weight: (<class ivy.data_classes.array.array.Array> shape=[128])
                },
                conv: {
                    weight: (<class ivy.data_classes.array.array.Array> shape=[128, 528, 1, 1])
                }
            }
        }
    },
    inception5a: {
        branch1: {
            bn: {
                bias: (<class ivy.data_classes.array.array.Array> shape=[256]),
                num_batches_tracked: ivy.array(0),
                running_mean: (<class ivy.data_classes.array.array.Array> shape=[256]),
                running_var: (<class ivy.data_classes.array.array.Array> shape=[256]),
                weight: (<class ivy.data_classes.array.array.Array> shape=[256])
            },
            conv: {
                weight: (<class ivy.data_classes.array.array.Array> shape=[256, 832, 1, 1])
            }
        },
        branch2: {
            0: {
                bn: {
                    bias: (<class ivy.data_classes.array.array.Array> shape=[160]),
                    num_batches_tracked: ivy.array(0),
                    running_mean: (<class ivy.data_classes.array.array.Array> shape=[160]),
                    running_var: (<class ivy.data_classes.array.array.Array> shape=[160]),
                    weight: (<class ivy.data_classes.array.array.Array> shape=[160])
                },
                conv: {
                    weight: (<class ivy.data_classes.array.array.Array> shape=[160, 832, 1, 1])
                }
            },
            1: {
                bn: {
                    bias: (<class ivy.data_classes.array.array.Array> shape=[320]),
                    num_batches_tracked: ivy.array(0),
                    running_mean: (<class ivy.data_classes.array.array.Array> shape=[320]),
                    running_var: (<class ivy.data_classes.array.array.Array> shape=[320]),
                    weight: (<class ivy.data_classes.array.array.Array> shape=[320])
                },
                conv: {
                    weight: (<class ivy.data_classes.array.array.Array> shape=[320, 160, 3, 3])
                }
            }
        },
        branch3: {
            0: {
                bn: {
                    bias: (<class ivy.data_classes.array.array.Array> shape=[32]),
                    num_batches_tracked: ivy.array(0),
                    running_mean: (<class ivy.data_classes.array.array.Array> shape=[32]),
                    running_var: (<class ivy.data_classes.array.array.Array> shape=[32]),
                    weight: (<class ivy.data_classes.array.array.Array> shape=[32])
                },
                conv: {
                    weight: (<class ivy.data_classes.array.array.Array> shape=[32, 832, 1, 1])
                }
            },
            1: {
                bn: {
                    bias: (<class ivy.data_classes.array.array.Array> shape=[128]),
                    num_batches_tracked: ivy.array(0),
                    running_mean: (<class ivy.data_classes.array.array.Array> shape=[128]),
                    running_var: (<class ivy.data_classes.array.array.Array> shape=[128]),
                    weight: (<class ivy.data_classes.array.array.Array> shape=[128])
                },
                conv: {
                    weight: (<class ivy.data_classes.array.array.Array> shape=[128, 32, 3, 3])
                }
            }
        },
        branch4: {
            1: {
                bn: {
                    bias: (<class ivy.data_classes.array.array.Array> shape=[128]),
                    num_batches_tracked: ivy.array(0),
                    running_mean: (<class ivy.data_classes.array.array.Array> shape=[128]),
                    running_var: (<class ivy.data_classes.array.array.Array> shape=[128]),
                    weight: (<class ivy.data_classes.array.array.Array> shape=[128])
                },
                conv: {
                    weight: (<class ivy.data_classes.array.array.Array> shape=[128, 832, 1, 1])
                }
            }
        }
    },
    inception5b: {
        branch1: {
            bn: {
                bias: (<class ivy.data_classes.array.array.Array> shape=[384]),
                num_batches_tracked: ivy.array(0),
                running_mean: (<class ivy.data_classes.array.array.Array> shape=[384]),
                running_var: (<class ivy.data_classes.array.array.Array> shape=[384]),
                weight: (<class ivy.data_classes.array.array.Array> shape=[384])
            },
            conv: {
                weight: (<class ivy.data_classes.array.array.Array> shape=[384, 832, 1, 1])
            }
        },
        branch2: {
            0: {
                bn: {
                    bias: (<class ivy.data_classes.array.array.Array> shape=[192]),
                    num_batches_tracked: ivy.array(0),
                    running_mean: (<class ivy.data_classes.array.array.Array> shape=[192]),
                    running_var: (<class ivy.data_classes.array.array.Array> shape=[192]),
                    weight: (<class ivy.data_classes.array.array.Array> shape=[192])
                },
                conv: {
                    weight: (<class ivy.data_classes.array.array.Array> shape=[192, 832, 1, 1])
                }
            },
            1: {
                bn: {
                    bias: (<class ivy.data_classes.array.array.Array> shape=[384]),
                    num_batches_tracked: ivy.array(0),
                    running_mean: (<class ivy.data_classes.array.array.Array> shape=[384]),
                    running_var: (<class ivy.data_classes.array.array.Array> shape=[384]),
                    weight: (<class ivy.data_classes.array.array.Array> shape=[384])
                },
                conv: {
                    weight: (<class ivy.data_classes.array.array.Array> shape=[384, 192, 3, 3])
                }
            }
        },
        branch3: {
            0: {
                bn: {
                    bias: (<class ivy.data_classes.array.array.Array> shape=[48]),
                    num_batches_tracked: ivy.array(0),
                    running_mean: (<class ivy.data_classes.array.array.Array> shape=[48]),
                    running_var: (<class ivy.data_classes.array.array.Array> shape=[48]),
                    weight: (<class ivy.data_classes.array.array.Array> shape=[48])
                },
                conv: {
                    weight: (<class ivy.data_classes.array.array.Array> shape=[48, 832, 1, 1])
                }
            },
            1: {
                bn: {
                    bias: (<class ivy.data_classes.array.array.Array> shape=[128]),
                    num_batches_tracked: ivy.array(0),
                    running_mean: (<class ivy.data_classes.array.array.Array> shape=[128]),
                    running_var: (<class ivy.data_classes.array.array.Array> shape=[128]),
                    weight: (<class ivy.data_classes.array.array.Array> shape=[128])
                },
                conv: {
                    weight: (<class ivy.data_classes.array.array.Array> shape=[128, 48, 3, 3])
                }
            }
        },
        branch4: {
            1: {
                bn: {
                    bias: (<class ivy.data_classes.array.array.Array> shape=[128]),
                    num_batches_tracked: ivy.array(0),
                    running_mean: (<class ivy.data_classes.array.array.Array> shape=[128]),
                    running_var: (<class ivy.data_classes.array.array.Array> shape=[128]),
                    weight: (<class ivy.data_classes.array.array.Array> shape=[128])
                },
                conv: {
                    weight: (<class ivy.data_classes.array.array.Array> shape=[128, 832, 1, 1])
                }
            }
        }
    }
}