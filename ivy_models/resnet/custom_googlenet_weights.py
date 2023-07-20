{
    aux4A: {
        conv: {
            w: (<class ivy.data_classes.array.array.Array> shape=[1, 1, 512, 128])
        },
        fc1: {
            b: (<class ivy.data_classes.array.array.Array> shape=[1024]),
            w: (<class ivy.data_classes.array.array.Array> shape=[1024, 2048])
        },
        fc2: {
            b: (<class ivy.data_classes.array.array.Array> shape=[1000]),
            w: (<class ivy.data_classes.array.array.Array> shape=[1000, 1024])
        }
    },
    aux4D: {
        conv: {
            w: (<class ivy.data_classes.array.array.Array> shape=[1, 1, 528, 128])
        },
        fc1: {
            b: (<class ivy.data_classes.array.array.Array> shape=[1024]),
            w: (<class ivy.data_classes.array.array.Array> shape=[1024, 2048])
        },
        fc2: {
            b: (<class ivy.data_classes.array.array.Array> shape=[1000]),
            w: (<class ivy.data_classes.array.array.Array> shape=[1000, 1024])
        }
    },
    conv1: {
        bn: {
            b: (<class ivy.data_classes.array.array.Array> shape=[64]),
            running_mean: (<class ivy.data_classes.array.array.Array> shape=[64]),
            running_var: (<class ivy.data_classes.array.array.Array> shape=[64]),
            w: (<class ivy.data_classes.array.array.Array> shape=[64])
        },
        conv: {
            w: (<class ivy.data_classes.array.array.Array> shape=[7, 7, 3, 64])
        }
    },
    conv2: {
        bn: {
            b: (<class ivy.data_classes.array.array.Array> shape=[64]),
            running_mean: (<class ivy.data_classes.array.array.Array> shape=[64]),
            running_var: (<class ivy.data_classes.array.array.Array> shape=[64]),
            w: (<class ivy.data_classes.array.array.Array> shape=[64])
        },
        conv: {
            w: (<class ivy.data_classes.array.array.Array> shape=[1, 1, 64, 64])
        }
    },
    conv3: {
        bn: {
            b: (<class ivy.data_classes.array.array.Array> shape=[192]),
            running_mean: (<class ivy.data_classes.array.array.Array> shape=[192]),
            running_var: (<class ivy.data_classes.array.array.Array> shape=[192]),
            w: (<class ivy.data_classes.array.array.Array> shape=[192])
        },
        conv: {
            w: (<class ivy.data_classes.array.array.Array> shape=[3, 3, 64, 192])
        }
    },
    fc: {
        b: (<class ivy.data_classes.array.array.Array> shape=[1000]),
        w: (<class ivy.data_classes.array.array.Array> shape=[1000, 1024])
    },
    inception3A: {
        block1: {
            submodules: {
                v0: {
                    bn: {
                        b: (<class ivy.data_classes.array.array.Array> shape=[64]),
                        running_mean: (<class ivy.data_classes.array.array.Array> shape=[64]),
                        running_var: (<class ivy.data_classes.array.array.Array> shape=[64]),
                        w: (<class ivy.data_classes.array.array.Array> shape=[64])
                    },
                    conv: {
                        w: (<class ivy.data_classes.array.array.Array> shape=[1, 1, 192, 64])
                    }
                }
            }
        },
        block2: {
            submodules: {
                v0: {
                    bn: {
                        b: (<class ivy.data_classes.array.array.Array> shape=[96]),
                        running_mean: (<class ivy.data_classes.array.array.Array> shape=[96]),
                        running_var: (<class ivy.data_classes.array.array.Array> shape=[96]),
                        w: (<class ivy.data_classes.array.array.Array> shape=[96])
                    },
                    conv: {
                        w: (<class ivy.data_classes.array.array.Array> shape=[1, 1, 192, 96])
                    }
                },
                v1: {
                    bn: {
                        b: (<class ivy.data_classes.array.array.Array> shape=[128]),
                        running_mean: (<class ivy.data_classes.array.array.Array> shape=[128]),
                        running_var: (<class ivy.data_classes.array.array.Array> shape=[128]),
                        w: (<class ivy.data_classes.array.array.Array> shape=[128])
                    },
                    conv: {
                        w: (<class ivy.data_classes.array.array.Array> shape=[3, 3, 96, 128])
                    }
                }
            }
        },
        block3: {
            submodules: {
                v0: {
                    bn: {
                        b: (<class ivy.data_classes.array.array.Array> shape=[16]),
                        running_mean: (<class ivy.data_classes.array.array.Array> shape=[16]),
                        running_var: (<class ivy.data_classes.array.array.Array> shape=[16]),
                        w: (<class ivy.data_classes.array.array.Array> shape=[16])
                    },
                    conv: {
                        w: (<class ivy.data_classes.array.array.Array> shape=[1, 1, 192, 16])
                    }
                },
                v1: {
                    bn: {
                        b: (<class ivy.data_classes.array.array.Array> shape=[32]),
                        running_mean: (<class ivy.data_classes.array.array.Array> shape=[32]),
                        running_var: (<class ivy.data_classes.array.array.Array> shape=[32]),
                        w: (<class ivy.data_classes.array.array.Array> shape=[32])
                    },
                    conv: {
                        w: (<class ivy.data_classes.array.array.Array> shape=[5, 5, 16, 32])
                    }
                }
            }
        },
        block4: {
            submodules: {
                v1: {
                    bn: {
                        b: (<class ivy.data_classes.array.array.Array> shape=[32]),
                        running_mean: (<class ivy.data_classes.array.array.Array> shape=[32]),
                        running_var: (<class ivy.data_classes.array.array.Array> shape=[32]),
                        w: (<class ivy.data_classes.array.array.Array> shape=[32])
                    },
                    conv: {
                        w: (<class ivy.data_classes.array.array.Array> shape=[1, 1, 192, 32])
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
                        running_mean: (<class ivy.data_classes.array.array.Array> shape=[128]),
                        running_var: (<class ivy.data_classes.array.array.Array> shape=[128]),
                        w: (<class ivy.data_classes.array.array.Array> shape=[128])
                    },
                    conv: {
                        w: (<class ivy.data_classes.array.array.Array> shape=[1, 1, 256, 128])
                    }
                }
            }
        },
        block2: {
            submodules: {
                v0: {
                    bn: {
                        b: (<class ivy.data_classes.array.array.Array> shape=[128]),
                        running_mean: (<class ivy.data_classes.array.array.Array> shape=[128]),
                        running_var: (<class ivy.data_classes.array.array.Array> shape=[128]),
                        w: (<class ivy.data_classes.array.array.Array> shape=[128])
                    },
                    conv: {
                        w: (<class ivy.data_classes.array.array.Array> shape=[1, 1, 256, 128])
                    }
                },
                v1: {
                    bn: {
                        b: (<class ivy.data_classes.array.array.Array> shape=[192]),
                        running_mean: (<class ivy.data_classes.array.array.Array> shape=[192]),
                        running_var: (<class ivy.data_classes.array.array.Array> shape=[192]),
                        w: (<class ivy.data_classes.array.array.Array> shape=[192])
                    },
                    conv: {
                        w: (<class ivy.data_classes.array.array.Array> shape=[3, 3, 128, 192])
                    }
                }
            }
        },
        block3: {
            submodules: {
                v0: {
                    bn: {
                        b: (<class ivy.data_classes.array.array.Array> shape=[32]),
                        running_mean: (<class ivy.data_classes.array.array.Array> shape=[32]),
                        running_var: (<class ivy.data_classes.array.array.Array> shape=[32]),
                        w: (<class ivy.data_classes.array.array.Array> shape=[32])
                    },
                    conv: {
                        w: (<class ivy.data_classes.array.array.Array> shape=[1, 1, 256, 32])
                    }
                },
                v1: {
                    bn: {
                        b: (<class ivy.data_classes.array.array.Array> shape=[96]),
                        running_mean: (<class ivy.data_classes.array.array.Array> shape=[96]),
                        running_var: (<class ivy.data_classes.array.array.Array> shape=[96]),
                        w: (<class ivy.data_classes.array.array.Array> shape=[96])
                    },
                    conv: {
                        w: (<class ivy.data_classes.array.array.Array> shape=[5, 5, 32, 96])
                    }
                }
            }
        },
        block4: {
            submodules: {
                v1: {
                    bn: {
                        b: (<class ivy.data_classes.array.array.Array> shape=[64]),
                        running_mean: (<class ivy.data_classes.array.array.Array> shape=[64]),
                        running_var: (<class ivy.data_classes.array.array.Array> shape=[64]),
                        w: (<class ivy.data_classes.array.array.Array> shape=[64])
                    },
                    conv: {
                        w: (<class ivy.data_classes.array.array.Array> shape=[1, 1, 256, 64])
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
                        b: (<class ivy.data_classes.array.array.Array> shape=[192]),
                        running_mean: (<class ivy.data_classes.array.array.Array> shape=[192]),
                        running_var: (<class ivy.data_classes.array.array.Array> shape=[192]),
                        w: (<class ivy.data_classes.array.array.Array> shape=[192])
                    },
                    conv: {
                        w: (<class ivy.data_classes.array.array.Array> shape=[1, 1, 480, 192])
                    }
                }
            }
        },
        block2: {
            submodules: {
                v0: {
                    bn: {
                        b: (<class ivy.data_classes.array.array.Array> shape=[96]),
                        running_mean: (<class ivy.data_classes.array.array.Array> shape=[96]),
                        running_var: (<class ivy.data_classes.array.array.Array> shape=[96]),
                        w: (<class ivy.data_classes.array.array.Array> shape=[96])
                    },
                    conv: {
                        w: (<class ivy.data_classes.array.array.Array> shape=[1, 1, 480, 96])
                    }
                },
                v1: {
                    bn: {
                        b: (<class ivy.data_classes.array.array.Array> shape=[208]),
                        running_mean: (<class ivy.data_classes.array.array.Array> shape=[208]),
                        running_var: (<class ivy.data_classes.array.array.Array> shape=[208]),
                        w: (<class ivy.data_classes.array.array.Array> shape=[208])
                    },
                    conv: {
                        w: (<class ivy.data_classes.array.array.Array> shape=[3, 3, 96, 208])
                    }
                }
            }
        },
        block3: {
            submodules: {
                v0: {
                    bn: {
                        b: (<class ivy.data_classes.array.array.Array> shape=[16]),
                        running_mean: (<class ivy.data_classes.array.array.Array> shape=[16]),
                        running_var: (<class ivy.data_classes.array.array.Array> shape=[16]),
                        w: (<class ivy.data_classes.array.array.Array> shape=[16])
                    },
                    conv: {
                        w: (<class ivy.data_classes.array.array.Array> shape=[1, 1, 480, 16])
                    }
                },
                v1: {
                    bn: {
                        b: (<class ivy.data_classes.array.array.Array> shape=[48]),
                        running_mean: (<class ivy.data_classes.array.array.Array> shape=[48]),
                        running_var: (<class ivy.data_classes.array.array.Array> shape=[48]),
                        w: (<class ivy.data_classes.array.array.Array> shape=[48])
                    },
                    conv: {
                        w: (<class ivy.data_classes.array.array.Array> shape=[5, 5, 16, 48])
                    }
                }
            }
        },
        block4: {
            submodules: {
                v1: {
                    bn: {
                        b: (<class ivy.data_classes.array.array.Array> shape=[64]),
                        running_mean: (<class ivy.data_classes.array.array.Array> shape=[64]),
                        running_var: (<class ivy.data_classes.array.array.Array> shape=[64]),
                        w: (<class ivy.data_classes.array.array.Array> shape=[64])
                    },
                    conv: {
                        w: (<class ivy.data_classes.array.array.Array> shape=[1, 1, 480, 64])
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
                        b: (<class ivy.data_classes.array.array.Array> shape=[160]),
                        running_mean: (<class ivy.data_classes.array.array.Array> shape=[160]),
                        running_var: (<class ivy.data_classes.array.array.Array> shape=[160]),
                        w: (<class ivy.data_classes.array.array.Array> shape=[160])
                    },
                    conv: {
                        w: (<class ivy.data_classes.array.array.Array> shape=[1, 1, 512, 160])
                    }
                }
            }
        },
        block2: {
            submodules: {
                v0: {
                    bn: {
                        b: (<class ivy.data_classes.array.array.Array> shape=[112]),
                        running_mean: (<class ivy.data_classes.array.array.Array> shape=[112]),
                        running_var: (<class ivy.data_classes.array.array.Array> shape=[112]),
                        w: (<class ivy.data_classes.array.array.Array> shape=[112])
                    },
                    conv: {
                        w: (<class ivy.data_classes.array.array.Array> shape=[1, 1, 512, 112])
                    }
                },
                v1: {
                    bn: {
                        b: (<class ivy.data_classes.array.array.Array> shape=[224]),
                        running_mean: (<class ivy.data_classes.array.array.Array> shape=[224]),
                        running_var: (<class ivy.data_classes.array.array.Array> shape=[224]),
                        w: (<class ivy.data_classes.array.array.Array> shape=[224])
                    },
                    conv: {
                        w: (<class ivy.data_classes.array.array.Array> shape=[3, 3, 112, 224])
                    }
                }
            }
        },
        block3: {
            submodules: {
                v0: {
                    bn: {
                        b: (<class ivy.data_classes.array.array.Array> shape=[24]),
                        running_mean: (<class ivy.data_classes.array.array.Array> shape=[24]),
                        running_var: (<class ivy.data_classes.array.array.Array> shape=[24]),
                        w: (<class ivy.data_classes.array.array.Array> shape=[24])
                    },
                    conv: {
                        w: (<class ivy.data_classes.array.array.Array> shape=[1, 1, 512, 24])
                    }
                },
                v1: {
                    bn: {
                        b: (<class ivy.data_classes.array.array.Array> shape=[64]),
                        running_mean: (<class ivy.data_classes.array.array.Array> shape=[64]),
                        running_var: (<class ivy.data_classes.array.array.Array> shape=[64]),
                        w: (<class ivy.data_classes.array.array.Array> shape=[64])
                    },
                    conv: {
                        w: (<class ivy.data_classes.array.array.Array> shape=[5, 5, 24, 64])
                    }
                }
            }
        },
        block4: {
            submodules: {
                v1: {
                    bn: {
                        b: (<class ivy.data_classes.array.array.Array> shape=[64]),
                        running_mean: (<class ivy.data_classes.array.array.Array> shape=[64]),
                        running_var: (<class ivy.data_classes.array.array.Array> shape=[64]),
                        w: (<class ivy.data_classes.array.array.Array> shape=[64])
                    },
                    conv: {
                        w: (<class ivy.data_classes.array.array.Array> shape=[1, 1, 512, 64])
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
                        running_mean: (<class ivy.data_classes.array.array.Array> shape=[128]),
                        running_var: (<class ivy.data_classes.array.array.Array> shape=[128]),
                        w: (<class ivy.data_classes.array.array.Array> shape=[128])
                    },
                    conv: {
                        w: (<class ivy.data_classes.array.array.Array> shape=[1, 1, 512, 128])
                    }
                }
            }
        },
        block2: {
            submodules: {
                v0: {
                    bn: {
                        b: (<class ivy.data_classes.array.array.Array> shape=[128]),
                        running_mean: (<class ivy.data_classes.array.array.Array> shape=[128]),
                        running_var: (<class ivy.data_classes.array.array.Array> shape=[128]),
                        w: (<class ivy.data_classes.array.array.Array> shape=[128])
                    },
                    conv: {
                        w: (<class ivy.data_classes.array.array.Array> shape=[1, 1, 512, 128])
                    }
                },
                v1: {
                    bn: {
                        b: (<class ivy.data_classes.array.array.Array> shape=[256]),
                        running_mean: (<class ivy.data_classes.array.array.Array> shape=[256]),
                        running_var: (<class ivy.data_classes.array.array.Array> shape=[256]),
                        w: (<class ivy.data_classes.array.array.Array> shape=[256])
                    },
                    conv: {
                        w: (<class ivy.data_classes.array.array.Array> shape=[3, 3, 128, 256])
                    }
                }
            }
        },
        block3: {
            submodules: {
                v0: {
                    bn: {
                        b: (<class ivy.data_classes.array.array.Array> shape=[24]),
                        running_mean: (<class ivy.data_classes.array.array.Array> shape=[24]),
                        running_var: (<class ivy.data_classes.array.array.Array> shape=[24]),
                        w: (<class ivy.data_classes.array.array.Array> shape=[24])
                    },
                    conv: {
                        w: (<class ivy.data_classes.array.array.Array> shape=[1, 1, 512, 24])
                    }
                },
                v1: {
                    bn: {
                        b: (<class ivy.data_classes.array.array.Array> shape=[64]),
                        running_mean: (<class ivy.data_classes.array.array.Array> shape=[64]),
                        running_var: (<class ivy.data_classes.array.array.Array> shape=[64]),
                        w: (<class ivy.data_classes.array.array.Array> shape=[64])
                    },
                    conv: {
                        w: (<class ivy.data_classes.array.array.Array> shape=[5, 5, 24, 64])
                    }
                }
            }
        },
        block4: {
            submodules: {
                v1: {
                    bn: {
                        b: (<class ivy.data_classes.array.array.Array> shape=[64]),
                        running_mean: (<class ivy.data_classes.array.array.Array> shape=[64]),
                        running_var: (<class ivy.data_classes.array.array.Array> shape=[64]),
                        w: (<class ivy.data_classes.array.array.Array> shape=[64])
                    },
                    conv: {
                        w: (<class ivy.data_classes.array.array.Array> shape=[1, 1, 512, 64])
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
                        b: (<class ivy.data_classes.array.array.Array> shape=[112]),
                        running_mean: (<class ivy.data_classes.array.array.Array> shape=[112]),
                        running_var: (<class ivy.data_classes.array.array.Array> shape=[112]),
                        w: (<class ivy.data_classes.array.array.Array> shape=[112])
                    },
                    conv: {
                        w: (<class ivy.data_classes.array.array.Array> shape=[1, 1, 512, 112])
                    }
                }
            }
        },
        block2: {
            submodules: {
                v0: {
                    bn: {
                        b: (<class ivy.data_classes.array.array.Array> shape=[144]),
                        running_mean: (<class ivy.data_classes.array.array.Array> shape=[144]),
                        running_var: (<class ivy.data_classes.array.array.Array> shape=[144]),
                        w: (<class ivy.data_classes.array.array.Array> shape=[144])
                    },
                    conv: {
                        w: (<class ivy.data_classes.array.array.Array> shape=[1, 1, 512, 144])
                    }
                },
                v1: {
                    bn: {
                        b: (<class ivy.data_classes.array.array.Array> shape=[288]),
                        running_mean: (<class ivy.data_classes.array.array.Array> shape=[288]),
                        running_var: (<class ivy.data_classes.array.array.Array> shape=[288]),
                        w: (<class ivy.data_classes.array.array.Array> shape=[288])
                    },
                    conv: {
                        w: (<class ivy.data_classes.array.array.Array> shape=[3, 3, 144, 288])
                    }
                }
            }
        },
        block3: {
            submodules: {
                v0: {
                    bn: {
                        b: (<class ivy.data_classes.array.array.Array> shape=[32]),
                        running_mean: (<class ivy.data_classes.array.array.Array> shape=[32]),
                        running_var: (<class ivy.data_classes.array.array.Array> shape=[32]),
                        w: (<class ivy.data_classes.array.array.Array> shape=[32])
                    },
                    conv: {
                        w: (<class ivy.data_classes.array.array.Array> shape=[1, 1, 512, 32])
                    }
                },
                v1: {
                    bn: {
                        b: (<class ivy.data_classes.array.array.Array> shape=[64]),
                        running_mean: (<class ivy.data_classes.array.array.Array> shape=[64]),
                        running_var: (<class ivy.data_classes.array.array.Array> shape=[64]),
                        w: (<class ivy.data_classes.array.array.Array> shape=[64])
                    },
                    conv: {
                        w: (<class ivy.data_classes.array.array.Array> shape=[5, 5, 32, 64])
                    }
                }
            }
        },
        block4: {
            submodules: {
                v1: {
                    bn: {
                        b: (<class ivy.data_classes.array.array.Array> shape=[64]),
                        running_mean: (<class ivy.data_classes.array.array.Array> shape=[64]),
                        running_var: (<class ivy.data_classes.array.array.Array> shape=[64]),
                        w: (<class ivy.data_classes.array.array.Array> shape=[64])
                    },
                    conv: {
                        w: (<class ivy.data_classes.array.array.Array> shape=[1, 1, 512, 64])
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
                        b: (<class ivy.data_classes.array.array.Array> shape=[256]),
                        running_mean: (<class ivy.data_classes.array.array.Array> shape=[256]),
                        running_var: (<class ivy.data_classes.array.array.Array> shape=[256]),
                        w: (<class ivy.data_classes.array.array.Array> shape=[256])
                    },
                    conv: {
                        w: (<class ivy.data_classes.array.array.Array> shape=[1, 1, 528, 256])
                    }
                }
            }
        },
        block2: {
            submodules: {
                v0: {
                    bn: {
                        b: (<class ivy.data_classes.array.array.Array> shape=[160]),
                        running_mean: (<class ivy.data_classes.array.array.Array> shape=[160]),
                        running_var: (<class ivy.data_classes.array.array.Array> shape=[160]),
                        w: (<class ivy.data_classes.array.array.Array> shape=[160])
                    },
                    conv: {
                        w: (<class ivy.data_classes.array.array.Array> shape=[1, 1, 528, 160])
                    }
                },
                v1: {
                    bn: {
                        b: (<class ivy.data_classes.array.array.Array> shape=[320]),
                        running_mean: (<class ivy.data_classes.array.array.Array> shape=[320]),
                        running_var: (<class ivy.data_classes.array.array.Array> shape=[320]),
                        w: (<class ivy.data_classes.array.array.Array> shape=[320])
                    },
                    conv: {
                        w: (<class ivy.data_classes.array.array.Array> shape=[3, 3, 160, 320])
                    }
                }
            }
        },
        block3: {
            submodules: {
                v0: {
                    bn: {
                        b: (<class ivy.data_classes.array.array.Array> shape=[32]),
                        running_mean: (<class ivy.data_classes.array.array.Array> shape=[32]),
                        running_var: (<class ivy.data_classes.array.array.Array> shape=[32]),
                        w: (<class ivy.data_classes.array.array.Array> shape=[32])
                    },
                    conv: {
                        w: (<class ivy.data_classes.array.array.Array> shape=[1, 1, 528, 32])
                    }
                },
                v1: {
                    bn: {
                        b: (<class ivy.data_classes.array.array.Array> shape=[128]),
                        running_mean: (<class ivy.data_classes.array.array.Array> shape=[128]),
                        running_var: (<class ivy.data_classes.array.array.Array> shape=[128]),
                        w: (<class ivy.data_classes.array.array.Array> shape=[128])
                    },
                    conv: {
                        w: (<class ivy.data_classes.array.array.Array> shape=[5, 5, 32, 128])
                    }
                }
            }
        },
        block4: {
            submodules: {
                v1: {
                    bn: {
                        b: (<class ivy.data_classes.array.array.Array> shape=[128]),
                        running_mean: (<class ivy.data_classes.array.array.Array> shape=[128]),
                        running_var: (<class ivy.data_classes.array.array.Array> shape=[128]),
                        w: (<class ivy.data_classes.array.array.Array> shape=[128])
                    },
                    conv: {
                        w: (<class ivy.data_classes.array.array.Array> shape=[1, 1, 528, 128])
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
                        b: (<class ivy.data_classes.array.array.Array> shape=[256]),
                        running_mean: (<class ivy.data_classes.array.array.Array> shape=[256]),
                        running_var: (<class ivy.data_classes.array.array.Array> shape=[256]),
                        w: (<class ivy.data_classes.array.array.Array> shape=[256])
                    },
                    conv: {
                        w: (<class ivy.data_classes.array.array.Array> shape=[1, 1, 832, 256])
                    }
                }
            }
        },
        block2: {
            submodules: {
                v0: {
                    bn: {
                        b: (<class ivy.data_classes.array.array.Array> shape=[160]),
                        running_mean: (<class ivy.data_classes.array.array.Array> shape=[160]),
                        running_var: (<class ivy.data_classes.array.array.Array> shape=[160]),
                        w: (<class ivy.data_classes.array.array.Array> shape=[160])
                    },
                    conv: {
                        w: (<class ivy.data_classes.array.array.Array> shape=[1, 1, 832, 160])
                    }
                },
                v1: {
                    bn: {
                        b: (<class ivy.data_classes.array.array.Array> shape=[320]),
                        running_mean: (<class ivy.data_classes.array.array.Array> shape=[320]),
                        running_var: (<class ivy.data_classes.array.array.Array> shape=[320]),
                        w: (<class ivy.data_classes.array.array.Array> shape=[320])
                    },
                    conv: {
                        w: (<class ivy.data_classes.array.array.Array> shape=[3, 3, 160, 320])
                    }
                }
            }
        },
        block3: {
            submodules: {
                v0: {
                    bn: {
                        b: (<class ivy.data_classes.array.array.Array> shape=[32]),
                        running_mean: (<class ivy.data_classes.array.array.Array> shape=[32]),
                        running_var: (<class ivy.data_classes.array.array.Array> shape=[32]),
                        w: (<class ivy.data_classes.array.array.Array> shape=[32])
                    },
                    conv: {
                        w: (<class ivy.data_classes.array.array.Array> shape=[1, 1, 832, 32])
                    }
                },
                v1: {
                    bn: {
                        b: (<class ivy.data_classes.array.array.Array> shape=[128]),
                        running_mean: (<class ivy.data_classes.array.array.Array> shape=[128]),
                        running_var: (<class ivy.data_classes.array.array.Array> shape=[128]),
                        w: (<class ivy.data_classes.array.array.Array> shape=[128])
                    },
                    conv: {
                        w: (<class ivy.data_classes.array.array.Array> shape=[5, 5, 32, 128])
                    }
                }
            }
        },
        block4: {
            submodules: {
                v1: {
                    bn: {
                        b: (<class ivy.data_classes.array.array.Array> shape=[128]),
                        running_mean: (<class ivy.data_classes.array.array.Array> shape=[128]),
                        running_var: (<class ivy.data_classes.array.array.Array> shape=[128]),
                        w: (<class ivy.data_classes.array.array.Array> shape=[128])
                    },
                    conv: {
                        w: (<class ivy.data_classes.array.array.Array> shape=[1, 1, 832, 128])
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
                        b: (<class ivy.data_classes.array.array.Array> shape=[384]),
                        running_mean: (<class ivy.data_classes.array.array.Array> shape=[384]),
                        running_var: (<class ivy.data_classes.array.array.Array> shape=[384]),
                        w: (<class ivy.data_classes.array.array.Array> shape=[384])
                    },
                    conv: {
                        w: (<class ivy.data_classes.array.array.Array> shape=[1, 1, 832, 384])
                    }
                }
            }
        },
        block2: {
            submodules: {
                v0: {
                    bn: {
                        b: (<class ivy.data_classes.array.array.Array> shape=[192]),
                        running_mean: (<class ivy.data_classes.array.array.Array> shape=[192]),
                        running_var: (<class ivy.data_classes.array.array.Array> shape=[192]),
                        w: (<class ivy.data_classes.array.array.Array> shape=[192])
                    },
                    conv: {
                        w: (<class ivy.data_classes.array.array.Array> shape=[1, 1, 832, 192])
                    }
                },
                v1: {
                    bn: {
                        b: (<class ivy.data_classes.array.array.Array> shape=[384]),
                        running_mean: (<class ivy.data_classes.array.array.Array> shape=[384]),
                        running_var: (<class ivy.data_classes.array.array.Array> shape=[384]),
                        w: (<class ivy.data_classes.array.array.Array> shape=[384])
                    },
                    conv: {
                        w: (<class ivy.data_classes.array.array.Array> shape=[3, 3, 192, 384])
                    }
                }
            }
        },
        block3: {
            submodules: {
                v0: {
                    bn: {
                        b: (<class ivy.data_classes.array.array.Array> shape=[48]),
                        running_mean: (<class ivy.data_classes.array.array.Array> shape=[48]),
                        running_var: (<class ivy.data_classes.array.array.Array> shape=[48]),
                        w: (<class ivy.data_classes.array.array.Array> shape=[48])
                    },
                    conv: {
                        w: (<class ivy.data_classes.array.array.Array> shape=[1, 1, 832, 48])
                    }
                },
                v1: {
                    bn: {
                        b: (<class ivy.data_classes.array.array.Array> shape=[128]),
                        running_mean: (<class ivy.data_classes.array.array.Array> shape=[128]),
                        running_var: (<class ivy.data_classes.array.array.Array> shape=[128]),
                        w: (<class ivy.data_classes.array.array.Array> shape=[128])
                    },
                    conv: {
                        w: (<class ivy.data_classes.array.array.Array> shape=[5, 5, 48, 128])
                    }
                }
            }
        },
        block4: {
            submodules: {
                v1: {
                    bn: {
                        b: (<class ivy.data_classes.array.array.Array> shape=[128]),
                        running_mean: (<class ivy.data_classes.array.array.Array> shape=[128]),
                        running_var: (<class ivy.data_classes.array.array.Array> shape=[128]),
                        w: (<class ivy.data_classes.array.array.Array> shape=[128])
                    },
                    conv: {
                        w: (<class ivy.data_classes.array.array.Array> shape=[1, 1, 832, 128])
                    }
                }
            }
        }
    }
}