{
    aux4A: {
        bn: {
            b: (<class ivy.data_classes.array.array.Array> shape=[128]),
            running_mean: (<class ivy.data_classes.array.array.Array> shape=[128]),
            running_var: (<class ivy.data_classes.array.array.Array> shape=[128]),
            w: (<class ivy.data_classes.array.array.Array> shape=[128])
        },
        conv: {
            bn: {
                b: (<class ivy.data_classes.array.array.Array> shape=[1, 1, 512, 128]),
                running_mean: (<class ivy.data_classes.array.array.Array> shape=[1024]),
                running_var: (<class ivy.data_classes.array.array.Array> shape=[1024, 2048]),
                w: (<class ivy.data_classes.array.array.Array> shape=[1000])
            },
            conv: {
                w: (<class ivy.data_classes.array.array.Array> shape=[1000, 1024])
            }
        },
        fc1: {
            b: (<class ivy.data_classes.array.array.Array> shape=[128]),
            w: (<class ivy.data_classes.array.array.Array> shape=[128])
        },
        fc2: {
            b: (<class ivy.data_classes.array.array.Array> shape=[128]),
            w: (<class ivy.data_classes.array.array.Array> shape=[128])
        }
    },
    aux4D: {
        bn: {
            b: (<class ivy.data_classes.array.array.Array> shape=[1, 1, 528, 128]),
            running_mean: (<class ivy.data_classes.array.array.Array> shape=[1024]),
            running_var: (<class ivy.data_classes.array.array.Array> shape=[1024, 2048]),
            w: (<class ivy.data_classes.array.array.Array> shape=[1000])
        },
        conv: {
            bn: {
                b: (<class ivy.data_classes.array.array.Array> shape=[1000, 1024]),
                running_mean: (<class ivy.data_classes.array.array.Array> shape=[64]),
                running_var: (<class ivy.data_classes.array.array.Array> shape=[64]),
                w: (<class ivy.data_classes.array.array.Array> shape=[64])
            },
            conv: {
                w: (<class ivy.data_classes.array.array.Array> shape=[64])
            }
        },
        fc1: {
            b: (<class ivy.data_classes.array.array.Array> shape=[7, 7, 3, 64]),
            w: (<class ivy.data_classes.array.array.Array> shape=[64])
        },
        fc2: {
            b: (<class ivy.data_classes.array.array.Array> shape=[64]),
            w: (<class ivy.data_classes.array.array.Array> shape=[64])
        }
    },
    conv1: {
        bn: {
            b: (<class ivy.data_classes.array.array.Array> shape=[64]),
            running_mean: (<class ivy.data_classes.array.array.Array> shape=[1, 1, 64, 64]),
            running_var: (<class ivy.data_classes.array.array.Array> shape=[192]),
            w: (<class ivy.data_classes.array.array.Array> shape=[192])
        },
        conv: {
            w: (<class ivy.data_classes.array.array.Array> shape=[192])
        }
    },
    conv2: {
        bn: {
            b: (<class ivy.data_classes.array.array.Array> shape=[192]),
            running_mean: (<class ivy.data_classes.array.array.Array> shape=[3, 3, 64, 192]),
            running_var: (<class ivy.data_classes.array.array.Array> shape=[1000]),
            w: (<class ivy.data_classes.array.array.Array> shape=[1000, 1024])
        },
        conv: {
            w: (<class ivy.data_classes.array.array.Array> shape=[64])
        }
    },
    conv3: {
        bn: {
            b: (<class ivy.data_classes.array.array.Array> shape=[64]),
            running_mean: (<class ivy.data_classes.array.array.Array> shape=[64]),
            running_var: (<class ivy.data_classes.array.array.Array> shape=[64]),
            w: (<class ivy.data_classes.array.array.Array> shape=[1, 1, 192, 64])
        },
        conv: {
            w: (<class ivy.data_classes.array.array.Array> shape=[96])
        }
    },
    fc: {
        b: (<class ivy.data_classes.array.array.Array> shape=[96]),
        w: (<class ivy.data_classes.array.array.Array> shape=[96])
    },
    inception3A: {
        block1: {
            submodules: {
                v0: {
                    bn: {
                        b: (<class ivy.data_classes.array.array.Array> shape=[96]),
                        running_mean: (<class ivy.data_classes.array.array.Array> shape=[1, 1, 192, 96]),
                        running_var: (<class ivy.data_classes.array.array.Array> shape=[128]),
                        w: (<class ivy.data_classes.array.array.Array> shape=[128])
                    },
                    conv: {
                        w: (<class ivy.data_classes.array.array.Array> shape=[128])
                    }
                }
            }
        },
        block2: {
            submodules: {
                v0: {
                    bn: {
                        b: (<class ivy.data_classes.array.array.Array> shape=[128]),
                        running_mean: (<class ivy.data_classes.array.array.Array> shape=[3, 3, 96, 128]),
                        running_var: (<class ivy.data_classes.array.array.Array> shape=[16]),
                        w: (<class ivy.data_classes.array.array.Array> shape=[16])
                    },
                    conv: {
                        w: (<class ivy.data_classes.array.array.Array> shape=[16])
                    }
                },
                v1: {
                    bn: {
                        b: (<class ivy.data_classes.array.array.Array> shape=[16]),
                        running_mean: (<class ivy.data_classes.array.array.Array> shape=[1, 1, 192, 16]),
                        running_var: (<class ivy.data_classes.array.array.Array> shape=[32]),
                        w: (<class ivy.data_classes.array.array.Array> shape=[32])
                    },
                    conv: {
                        w: (<class ivy.data_classes.array.array.Array> shape=[32])
                    }
                }
            }
        },
        block3: {
            submodules: {
                v0: {
                    bn: {
                        b: (<class ivy.data_classes.array.array.Array> shape=[32]),
                        running_mean: (<class ivy.data_classes.array.array.Array> shape=[3, 3, 16, 32]),
                        running_var: (<class ivy.data_classes.array.array.Array> shape=[32]),
                        w: (<class ivy.data_classes.array.array.Array> shape=[32])
                    },
                    conv: {
                        w: (<class ivy.data_classes.array.array.Array> shape=[32])
                    }
                },
                v1: {
                    bn: {
                        b: (<class ivy.data_classes.array.array.Array> shape=[32]),
                        running_mean: (<class ivy.data_classes.array.array.Array> shape=[1, 1, 192, 32]),
                        running_var: (<class ivy.data_classes.array.array.Array> shape=[128]),
                        w: (<class ivy.data_classes.array.array.Array> shape=[128])
                    },
                    conv: {
                        w: (<class ivy.data_classes.array.array.Array> shape=[128])
                    }
                }
            }
        },
        block4: {
            submodules: {
                v1: {
                    bn: {
                        b: (<class ivy.data_classes.array.array.Array> shape=[128]),
                        running_mean: (<class ivy.data_classes.array.array.Array> shape=[1, 1, 256, 128]),
                        running_var: (<class ivy.data_classes.array.array.Array> shape=[128]),
                        w: (<class ivy.data_classes.array.array.Array> shape=[128])
                    },
                    conv: {
                        w: (<class ivy.data_classes.array.array.Array> shape=[128])
                    }
                }
            }
        }
    },
    inception3B: {
        block1: {
            submodules: {
                v0: {
                    bn: {
                        b: (<class ivy.data_classes.array.array.Array> shape=[128]),
                        running_mean: (<class ivy.data_classes.array.array.Array> shape=[1, 1, 256, 128]),
                        running_var: (<class ivy.data_classes.array.array.Array> shape=[192]),
                        w: (<class ivy.data_classes.array.array.Array> shape=[192])
                    },
                    conv: {
                        w: (<class ivy.data_classes.array.array.Array> shape=[192])
                    }
                }
            }
        },
        block2: {
            submodules: {
                v0: {
                    bn: {
                        b: (<class ivy.data_classes.array.array.Array> shape=[192]),
                        running_mean: (<class ivy.data_classes.array.array.Array> shape=[3, 3, 128, 192]),
                        running_var: (<class ivy.data_classes.array.array.Array> shape=[32]),
                        w: (<class ivy.data_classes.array.array.Array> shape=[32])
                    },
                    conv: {
                        w: (<class ivy.data_classes.array.array.Array> shape=[32])
                    }
                },
                v1: {
                    bn: {
                        b: (<class ivy.data_classes.array.array.Array> shape=[32]),
                        running_mean: (<class ivy.data_classes.array.array.Array> shape=[1, 1, 256, 32]),
                        running_var: (<class ivy.data_classes.array.array.Array> shape=[96]),
                        w: (<class ivy.data_classes.array.array.Array> shape=[96])
                    },
                    conv: {
                        w: (<class ivy.data_classes.array.array.Array> shape=[96])
                    }
                }
            }
        },
        block3: {
            submodules: {
                v0: {
                    bn: {
                        b: (<class ivy.data_classes.array.array.Array> shape=[96]),
                        running_mean: (<class ivy.data_classes.array.array.Array> shape=[3, 3, 32, 96]),
                        running_var: (<class ivy.data_classes.array.array.Array> shape=[64]),
                        w: (<class ivy.data_classes.array.array.Array> shape=[64])
                    },
                    conv: {
                        w: (<class ivy.data_classes.array.array.Array> shape=[64])
                    }
                },
                v1: {
                    bn: {
                        b: (<class ivy.data_classes.array.array.Array> shape=[64]),
                        running_mean: (<class ivy.data_classes.array.array.Array> shape=[1, 1, 256, 64]),
                        running_var: (<class ivy.data_classes.array.array.Array> shape=[192]),
                        w: (<class ivy.data_classes.array.array.Array> shape=[192])
                    },
                    conv: {
                        w: (<class ivy.data_classes.array.array.Array> shape=[192])
                    }
                }
            }
        },
        block4: {
            submodules: {
                v1: {
                    bn: {
                        b: (<class ivy.data_classes.array.array.Array> shape=[192]),
                        running_mean: (<class ivy.data_classes.array.array.Array> shape=[1, 1, 480, 192]),
                        running_var: (<class ivy.data_classes.array.array.Array> shape=[96]),
                        w: (<class ivy.data_classes.array.array.Array> shape=[96])
                    },
                    conv: {
                        w: (<class ivy.data_classes.array.array.Array> shape=[96])
                    }
                }
            }
        }
    },
    inception4A: {
        block1: {
            submodules: {
                v0: {
                    bn: {
                        b: (<class ivy.data_classes.array.array.Array> shape=[96]),
                        running_mean: (<class ivy.data_classes.array.array.Array> shape=[1, 1, 480, 96]),
                        running_var: (<class ivy.data_classes.array.array.Array> shape=[208]),
                        w: (<class ivy.data_classes.array.array.Array> shape=[208])
                    },
                    conv: {
                        w: (<class ivy.data_classes.array.array.Array> shape=[208])
                    }
                }
            }
        },
        block2: {
            submodules: {
                v0: {
                    bn: {
                        b: (<class ivy.data_classes.array.array.Array> shape=[208]),
                        running_mean: (<class ivy.data_classes.array.array.Array> shape=[3, 3, 96, 208]),
                        running_var: (<class ivy.data_classes.array.array.Array> shape=[16]),
                        w: (<class ivy.data_classes.array.array.Array> shape=[16])
                    },
                    conv: {
                        w: (<class ivy.data_classes.array.array.Array> shape=[16])
                    }
                },
                v1: {
                    bn: {
                        b: (<class ivy.data_classes.array.array.Array> shape=[16]),
                        running_mean: (<class ivy.data_classes.array.array.Array> shape=[1, 1, 480, 16]),
                        running_var: (<class ivy.data_classes.array.array.Array> shape=[48]),
                        w: (<class ivy.data_classes.array.array.Array> shape=[48])
                    },
                    conv: {
                        w: (<class ivy.data_classes.array.array.Array> shape=[48])
                    }
                }
            }
        },
        block3: {
            submodules: {
                v0: {
                    bn: {
                        b: (<class ivy.data_classes.array.array.Array> shape=[48]),
                        running_mean: (<class ivy.data_classes.array.array.Array> shape=[3, 3, 16, 48]),
                        running_var: (<class ivy.data_classes.array.array.Array> shape=[64]),
                        w: (<class ivy.data_classes.array.array.Array> shape=[64])
                    },
                    conv: {
                        w: (<class ivy.data_classes.array.array.Array> shape=[64])
                    }
                },
                v1: {
                    bn: {
                        b: (<class ivy.data_classes.array.array.Array> shape=[64]),
                        running_mean: (<class ivy.data_classes.array.array.Array> shape=[1, 1, 480, 64]),
                        running_var: (<class ivy.data_classes.array.array.Array> shape=[160]),
                        w: (<class ivy.data_classes.array.array.Array> shape=[160])
                    },
                    conv: {
                        w: (<class ivy.data_classes.array.array.Array> shape=[160])
                    }
                }
            }
        },
        block4: {
            submodules: {
                v1: {
                    bn: {
                        b: (<class ivy.data_classes.array.array.Array> shape=[160]),
                        running_mean: (<class ivy.data_classes.array.array.Array> shape=[1, 1, 512, 160]),
                        running_var: (<class ivy.data_classes.array.array.Array> shape=[112]),
                        w: (<class ivy.data_classes.array.array.Array> shape=[112])
                    },
                    conv: {
                        w: (<class ivy.data_classes.array.array.Array> shape=[112])
                    }
                }
            }
        }
    },
    inception4B: {
        block1: {
            submodules: {
                v0: {
                    bn: {
                        b: (<class ivy.data_classes.array.array.Array> shape=[112]),
                        running_mean: (<class ivy.data_classes.array.array.Array> shape=[1, 1, 512, 112]),
                        running_var: (<class ivy.data_classes.array.array.Array> shape=[224]),
                        w: (<class ivy.data_classes.array.array.Array> shape=[224])
                    },
                    conv: {
                        w: (<class ivy.data_classes.array.array.Array> shape=[224])
                    }
                }
            }
        },
        block2: {
            submodules: {
                v0: {
                    bn: {
                        b: (<class ivy.data_classes.array.array.Array> shape=[224]),
                        running_mean: (<class ivy.data_classes.array.array.Array> shape=[3, 3, 112, 224]),
                        running_var: (<class ivy.data_classes.array.array.Array> shape=[24]),
                        w: (<class ivy.data_classes.array.array.Array> shape=[24])
                    },
                    conv: {
                        w: (<class ivy.data_classes.array.array.Array> shape=[24])
                    }
                },
                v1: {
                    bn: {
                        b: (<class ivy.data_classes.array.array.Array> shape=[24]),
                        running_mean: (<class ivy.data_classes.array.array.Array> shape=[1, 1, 512, 24]),
                        running_var: (<class ivy.data_classes.array.array.Array> shape=[64]),
                        w: (<class ivy.data_classes.array.array.Array> shape=[64])
                    },
                    conv: {
                        w: (<class ivy.data_classes.array.array.Array> shape=[64])
                    }
                }
            }
        },
        block3: {
            submodules: {
                v0: {
                    bn: {
                        b: (<class ivy.data_classes.array.array.Array> shape=[64]),
                        running_mean: (<class ivy.data_classes.array.array.Array> shape=[3, 3, 24, 64]),
                        running_var: (<class ivy.data_classes.array.array.Array> shape=[64]),
                        w: (<class ivy.data_classes.array.array.Array> shape=[64])
                    },
                    conv: {
                        w: (<class ivy.data_classes.array.array.Array> shape=[64])
                    }
                },
                v1: {
                    bn: {
                        b: (<class ivy.data_classes.array.array.Array> shape=[64]),
                        running_mean: (<class ivy.data_classes.array.array.Array> shape=[1, 1, 512, 64]),
                        running_var: (<class ivy.data_classes.array.array.Array> shape=[128]),
                        w: (<class ivy.data_classes.array.array.Array> shape=[128])
                    },
                    conv: {
                        w: (<class ivy.data_classes.array.array.Array> shape=[128])
                    }
                }
            }
        },
        block4: {
            submodules: {
                v1: {
                    bn: {
                        b: (<class ivy.data_classes.array.array.Array> shape=[128]),
                        running_mean: (<class ivy.data_classes.array.array.Array> shape=[1, 1, 512, 128]),
                        running_var: (<class ivy.data_classes.array.array.Array> shape=[128]),
                        w: (<class ivy.data_classes.array.array.Array> shape=[128])
                    },
                    conv: {
                        w: (<class ivy.data_classes.array.array.Array> shape=[128])
                    }
                }
            }
        }
    },
    inception4C: {
        block1: {
            submodules: {
                v0: {
                    bn: {
                        b: (<class ivy.data_classes.array.array.Array> shape=[128]),
                        running_mean: (<class ivy.data_classes.array.array.Array> shape=[1, 1, 512, 128]),
                        running_var: (<class ivy.data_classes.array.array.Array> shape=[256]),
                        w: (<class ivy.data_classes.array.array.Array> shape=[256])
                    },
                    conv: {
                        w: (<class ivy.data_classes.array.array.Array> shape=[256])
                    }
                }
            }
        },
        block2: {
            submodules: {
                v0: {
                    bn: {
                        b: (<class ivy.data_classes.array.array.Array> shape=[256]),
                        running_mean: (<class ivy.data_classes.array.array.Array> shape=[3, 3, 128, 256]),
                        running_var: (<class ivy.data_classes.array.array.Array> shape=[24]),
                        w: (<class ivy.data_classes.array.array.Array> shape=[24])
                    },
                    conv: {
                        w: (<class ivy.data_classes.array.array.Array> shape=[24])
                    }
                },
                v1: {
                    bn: {
                        b: (<class ivy.data_classes.array.array.Array> shape=[24]),
                        running_mean: (<class ivy.data_classes.array.array.Array> shape=[1, 1, 512, 24]),
                        running_var: (<class ivy.data_classes.array.array.Array> shape=[64]),
                        w: (<class ivy.data_classes.array.array.Array> shape=[64])
                    },
                    conv: {
                        w: (<class ivy.data_classes.array.array.Array> shape=[64])
                    }
                }
            }
        },
        block3: {
            submodules: {
                v0: {
                    bn: {
                        b: (<class ivy.data_classes.array.array.Array> shape=[64]),
                        running_mean: (<class ivy.data_classes.array.array.Array> shape=[3, 3, 24, 64]),
                        running_var: (<class ivy.data_classes.array.array.Array> shape=[64]),
                        w: (<class ivy.data_classes.array.array.Array> shape=[64])
                    },
                    conv: {
                        w: (<class ivy.data_classes.array.array.Array> shape=[64])
                    }
                },
                v1: {
                    bn: {
                        b: (<class ivy.data_classes.array.array.Array> shape=[64]),
                        running_mean: (<class ivy.data_classes.array.array.Array> shape=[1, 1, 512, 64]),
                        running_var: (<class ivy.data_classes.array.array.Array> shape=[112]),
                        w: (<class ivy.data_classes.array.array.Array> shape=[112])
                    },
                    conv: {
                        w: (<class ivy.data_classes.array.array.Array> shape=[112])
                    }
                }
            }
        },
        block4: {
            submodules: {
                v1: {
                    bn: {
                        b: (<class ivy.data_classes.array.array.Array> shape=[112]),
                        running_mean: (<class ivy.data_classes.array.array.Array> shape=[1, 1, 512, 112]),
                        running_var: (<class ivy.data_classes.array.array.Array> shape=[144]),
                        w: (<class ivy.data_classes.array.array.Array> shape=[144])
                    },
                    conv: {
                        w: (<class ivy.data_classes.array.array.Array> shape=[144])
                    }
                }
            }
        }
    },
    inception4D: {
        block1: {
            submodules: {
                v0: {
                    bn: {
                        b: (<class ivy.data_classes.array.array.Array> shape=[144]),
                        running_mean: (<class ivy.data_classes.array.array.Array> shape=[1, 1, 512, 144]),
                        running_var: (<class ivy.data_classes.array.array.Array> shape=[288]),
                        w: (<class ivy.data_classes.array.array.Array> shape=[288])
                    },
                    conv: {
                        w: (<class ivy.data_classes.array.array.Array> shape=[288])
                    }
                }
            }
        },
        block2: {
            submodules: {
                v0: {
                    bn: {
                        b: (<class ivy.data_classes.array.array.Array> shape=[288]),
                        running_mean: (<class ivy.data_classes.array.array.Array> shape=[3, 3, 144, 288]),
                        running_var: (<class ivy.data_classes.array.array.Array> shape=[32]),
                        w: (<class ivy.data_classes.array.array.Array> shape=[32])
                    },
                    conv: {
                        w: (<class ivy.data_classes.array.array.Array> shape=[32])
                    }
                },
                v1: {
                    bn: {
                        b: (<class ivy.data_classes.array.array.Array> shape=[32]),
                        running_mean: (<class ivy.data_classes.array.array.Array> shape=[1, 1, 512, 32]),
                        running_var: (<class ivy.data_classes.array.array.Array> shape=[64]),
                        w: (<class ivy.data_classes.array.array.Array> shape=[64])
                    },
                    conv: {
                        w: (<class ivy.data_classes.array.array.Array> shape=[64])
                    }
                }
            }
        },
        block3: {
            submodules: {
                v0: {
                    bn: {
                        b: (<class ivy.data_classes.array.array.Array> shape=[64]),
                        running_mean: (<class ivy.data_classes.array.array.Array> shape=[3, 3, 32, 64]),
                        running_var: (<class ivy.data_classes.array.array.Array> shape=[64]),
                        w: (<class ivy.data_classes.array.array.Array> shape=[64])
                    },
                    conv: {
                        w: (<class ivy.data_classes.array.array.Array> shape=[64])
                    }
                },
                v1: {
                    bn: {
                        b: (<class ivy.data_classes.array.array.Array> shape=[64]),
                        running_mean: (<class ivy.data_classes.array.array.Array> shape=[1, 1, 512, 64]),
                        running_var: (<class ivy.data_classes.array.array.Array> shape=[256]),
                        w: (<class ivy.data_classes.array.array.Array> shape=[256])
                    },
                    conv: {
                        w: (<class ivy.data_classes.array.array.Array> shape=[256])
                    }
                }
            }
        },
        block4: {
            submodules: {
                v1: {
                    bn: {
                        b: (<class ivy.data_classes.array.array.Array> shape=[256]),
                        running_mean: (<class ivy.data_classes.array.array.Array> shape=[1, 1, 528, 256]),
                        running_var: (<class ivy.data_classes.array.array.Array> shape=[160]),
                        w: (<class ivy.data_classes.array.array.Array> shape=[160])
                    },
                    conv: {
                        w: (<class ivy.data_classes.array.array.Array> shape=[160])
                    }
                }
            }
        }
    },
    inception4E: {
        block1: {
            submodules: {
                v0: {
                    bn: {
                        b: (<class ivy.data_classes.array.array.Array> shape=[160]),
                        running_mean: (<class ivy.data_classes.array.array.Array> shape=[1, 1, 528, 160]),
                        running_var: (<class ivy.data_classes.array.array.Array> shape=[320]),
                        w: (<class ivy.data_classes.array.array.Array> shape=[320])
                    },
                    conv: {
                        w: (<class ivy.data_classes.array.array.Array> shape=[320])
                    }
                }
            }
        },
        block2: {
            submodules: {
                v0: {
                    bn: {
                        b: (<class ivy.data_classes.array.array.Array> shape=[320]),
                        running_mean: (<class ivy.data_classes.array.array.Array> shape=[3, 3, 160, 320]),
                        running_var: (<class ivy.data_classes.array.array.Array> shape=[32]),
                        w: (<class ivy.data_classes.array.array.Array> shape=[32])
                    },
                    conv: {
                        w: (<class ivy.data_classes.array.array.Array> shape=[32])
                    }
                },
                v1: {
                    bn: {
                        b: (<class ivy.data_classes.array.array.Array> shape=[32]),
                        running_mean: (<class ivy.data_classes.array.array.Array> shape=[1, 1, 528, 32]),
                        running_var: (<class ivy.data_classes.array.array.Array> shape=[128]),
                        w: (<class ivy.data_classes.array.array.Array> shape=[128])
                    },
                    conv: {
                        w: (<class ivy.data_classes.array.array.Array> shape=[128])
                    }
                }
            }
        },
        block3: {
            submodules: {
                v0: {
                    bn: {
                        b: (<class ivy.data_classes.array.array.Array> shape=[128]),
                        running_mean: (<class ivy.data_classes.array.array.Array> shape=[3, 3, 32, 128]),
                        running_var: (<class ivy.data_classes.array.array.Array> shape=[128]),
                        w: (<class ivy.data_classes.array.array.Array> shape=[128])
                    },
                    conv: {
                        w: (<class ivy.data_classes.array.array.Array> shape=[128])
                    }
                },
                v1: {
                    bn: {
                        b: (<class ivy.data_classes.array.array.Array> shape=[128]),
                        running_mean: (<class ivy.data_classes.array.array.Array> shape=[1, 1, 528, 128]),
                        running_var: (<class ivy.data_classes.array.array.Array> shape=[256]),
                        w: (<class ivy.data_classes.array.array.Array> shape=[256])
                    },
                    conv: {
                        w: (<class ivy.data_classes.array.array.Array> shape=[256])
                    }
                }
            }
        },
        block4: {
            submodules: {
                v1: {
                    bn: {
                        b: (<class ivy.data_classes.array.array.Array> shape=[256]),
                        running_mean: (<class ivy.data_classes.array.array.Array> shape=[1, 1, 832, 256]),
                        running_var: (<class ivy.data_classes.array.array.Array> shape=[160]),
                        w: (<class ivy.data_classes.array.array.Array> shape=[160])
                    },
                    conv: {
                        w: (<class ivy.data_classes.array.array.Array> shape=[160])
                    }
                }
            }
        }
    },
    inception5A: {
        block1: {
            submodules: {
                v0: {
                    bn: {
                        b: (<class ivy.data_classes.array.array.Array> shape=[160]),
                        running_mean: (<class ivy.data_classes.array.array.Array> shape=[1, 1, 832, 160]),
                        running_var: (<class ivy.data_classes.array.array.Array> shape=[320]),
                        w: (<class ivy.data_classes.array.array.Array> shape=[320])
                    },
                    conv: {
                        w: (<class ivy.data_classes.array.array.Array> shape=[320])
                    }
                }
            }
        },
        block2: {
            submodules: {
                v0: {
                    bn: {
                        b: (<class ivy.data_classes.array.array.Array> shape=[320]),
                        running_mean: (<class ivy.data_classes.array.array.Array> shape=[3, 3, 160, 320]),
                        running_var: (<class ivy.data_classes.array.array.Array> shape=[32]),
                        w: (<class ivy.data_classes.array.array.Array> shape=[32])
                    },
                    conv: {
                        w: (<class ivy.data_classes.array.array.Array> shape=[32])
                    }
                },
                v1: {
                    bn: {
                        b: (<class ivy.data_classes.array.array.Array> shape=[32]),
                        running_mean: (<class ivy.data_classes.array.array.Array> shape=[1, 1, 832, 32]),
                        running_var: (<class ivy.data_classes.array.array.Array> shape=[128]),
                        w: (<class ivy.data_classes.array.array.Array> shape=[128])
                    },
                    conv: {
                        w: (<class ivy.data_classes.array.array.Array> shape=[128])
                    }
                }
            }
        },
        block3: {
            submodules: {
                v0: {
                    bn: {
                        b: (<class ivy.data_classes.array.array.Array> shape=[128]),
                        running_mean: (<class ivy.data_classes.array.array.Array> shape=[3, 3, 32, 128]),
                        running_var: (<class ivy.data_classes.array.array.Array> shape=[128]),
                        w: (<class ivy.data_classes.array.array.Array> shape=[128])
                    },
                    conv: {
                        w: (<class ivy.data_classes.array.array.Array> shape=[128])
                    }
                },
                v1: {
                    bn: {
                        b: (<class ivy.data_classes.array.array.Array> shape=[128]),
                        running_mean: (<class ivy.data_classes.array.array.Array> shape=[1, 1, 832, 128]),
                        running_var: (<class ivy.data_classes.array.array.Array> shape=[384]),
                        w: (<class ivy.data_classes.array.array.Array> shape=[384])
                    },
                    conv: {
                        w: (<class ivy.data_classes.array.array.Array> shape=[384])
                    }
                }
            }
        },
        block4: {
            submodules: {
                v1: {
                    bn: {
                        b: (<class ivy.data_classes.array.array.Array> shape=[384]),
                        running_mean: (<class ivy.data_classes.array.array.Array> shape=[1, 1, 832, 384]),
                        running_var: (<class ivy.data_classes.array.array.Array> shape=[192]),
                        w: (<class ivy.data_classes.array.array.Array> shape=[192])
                    },
                    conv: {
                        w: (<class ivy.data_classes.array.array.Array> shape=[192])
                    }
                }
            }
        }
    },
    inception5B: {
        block1: {
            submodules: {
                v0: {
                    bn: {
                        b: (<class ivy.data_classes.array.array.Array> shape=[192]),
                        running_mean: (<class ivy.data_classes.array.array.Array> shape=[1, 1, 832, 192]),
                        running_var: (<class ivy.data_classes.array.array.Array> shape=[384]),
                        w: (<class ivy.data_classes.array.array.Array> shape=[384])
                    },
                    conv: {
                        w: (<class ivy.data_classes.array.array.Array> shape=[384])
                    }
                }
            }
        },
        block2: {
            submodules: {
                v0: {
                    bn: {
                        b: (<class ivy.data_classes.array.array.Array> shape=[384]),
                        running_mean: (<class ivy.data_classes.array.array.Array> shape=[3, 3, 192, 384]),
                        running_var: (<class ivy.data_classes.array.array.Array> shape=[48]),
                        w: (<class ivy.data_classes.array.array.Array> shape=[48])
                    },
                    conv: {
                        w: (<class ivy.data_classes.array.array.Array> shape=[48])
                    }
                },
                v1: {
                    bn: {
                        b: (<class ivy.data_classes.array.array.Array> shape=[48]),
                        running_mean: (<class ivy.data_classes.array.array.Array> shape=[1, 1, 832, 48]),
                        running_var: (<class ivy.data_classes.array.array.Array> shape=[128]),
                        w: (<class ivy.data_classes.array.array.Array> shape=[128])
                    },
                    conv: {
                        w: (<class ivy.data_classes.array.array.Array> shape=[128])
                    }
                }
            }
        },
        block3: {
            submodules: {
                v0: {
                    bn: {
                        b: (<class ivy.data_classes.array.array.Array> shape=[128]),
                        running_mean: (<class ivy.data_classes.array.array.Array> shape=[3, 3, 48, 128]),
                        running_var: (<class ivy.data_classes.array.array.Array> shape=[128]),
                        w: (<class ivy.data_classes.array.array.Array> shape=[128])
                    },
                    conv: {
                        w: (<class ivy.data_classes.array.array.Array> shape=[128])
                    }
                },
                v1: {
                    bn: {
                        b: (<class ivy.data_classes.array.array.Array> shape=[128]),
                        running_mean: (<class ivy.data_classes.array.array.Array> shape=[1, 1, 832, 128])
                    }
                }
            }
        }
    }
}