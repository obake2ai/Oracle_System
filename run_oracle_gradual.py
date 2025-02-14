def stylegan_frame_generator(frame_queue, stop_event, config_args, batch_size=4):
    device = torch.device(config_args["stylegan_gpu"])
    noise_seed = config_args["noise_seed"]
    torch.manual_seed(noise_seed)
    np.random.seed(noise_seed)
    random.seed(noise_seed)
    os.makedirs(config_args["out_dir"], exist_ok=True)
    Gs_kwargs = dnnlib.EasyDict()
    for key in ["verbose", "size", "scale_type"]:
        Gs_kwargs[key] = config_args[key]

    # ...（latmask, dconst, trans_params の初期化はそのまま）
    if config_args["latmask"] is None:
        nxy = config_args["nXY"]
        nHW = [int(s) for s in nxy.split('-')][::-1]
        n_mult = nHW[0] * nHW[1]
        if config_args["splitmax"] is not None:
            n_mult = min(n_mult, config_args["splitmax"])
        if config_args["verbose"] and n_mult > 1:
            print(f"Latent blending with split frame {nHW[1]} x {nHW[0]}")
        Gs_kwargs.countHW = nHW
        Gs_kwargs.splitfine = config_args["splitfine"]
        lmask = [None]
    else:
        # ...（略：既存コードそのまま）
        pass

    frames_val, fstep_val = [int(x) for x in config_args["frames"].split('-')]
    model_path = config_args["model"]
    pkl_name = os.path.splitext(model_path)[0]
    custom = False if '.pkl' in model_path.lower() else True
    with dnnlib.util.open_url(pkl_name + '.pkl') as f:
        rot = True if ('-r-' in model_path.lower() or 'sg3r-' in model_path.lower()) else False
        Gs = legacy.load_network_pkl(f, custom=custom, rot=rot, **Gs_kwargs)['G_ema'].to(device)
    z_dim = Gs.z_dim
    c_dim = Gs.c_dim
    if c_dim > 0 and config_args["labels"] is not None:
        label = torch.zeros([1, c_dim], device=device)
        label_idx = min(int(config_args["labels"]), c_dim - 1)
        label[0, label_idx] = 1
    else:
        label = None

    # 既に一度 forward のウォームアップ済みならそのまま
    if hasattr(Gs.synthesis, 'input'):
        # (内部パラメータの初期化はそのまま)
        first_layer_channels = Gs.synthesis.input.channels
        first_layer_size = Gs.synthesis.input.size
        if isinstance(first_layer_size, (list, tuple, np.ndarray)):
            h, w = first_layer_size[0], first_layer_size[1]
        else:
            h, w = first_layer_size, first_layer_size
        shape_for_dconst = [1, first_layer_channels, h, w]
        if config_args["digress"] != 0:
            dconst_list = []
            for i in range(n_mult):
                dc_tmp = config_args["digress"] * latent_anima(shape_for_dconst, frames_val, fstep_val,
                                                                 cubic=True, seed=noise_seed, verbose=False)
                dconst_list.append(dc_tmp)
            dconst = np.concatenate(dconst_list, axis=1)
        else:
            dconst = np.zeros([shifts.shape[0], 1, first_layer_channels, h, w])
        dconst = torch.from_numpy(dconst).to(device).to(torch.float32)
    else:
        dconst = None

    # ここからバッチ処理へ変更
    frame_idx_local = 0
    frame_idx = 0
    if config_args["method"] == "random_walk":
        print("random")
        latent_gen = infinite_latent_random_walk(z_dim=z_dim, device=device, seed=noise_seed, step_size=0.02)
    else:
        print("smooth")
        latent_gen = infinite_latent_smooth(z_dim=z_dim, device=device,
                                            cubic=config_args["cubic"],
                                            gauss=config_args["gauss"],
                                            seed=noise_seed,
                                            chunk_size=config_args["chunk_size"],
                                            uniform=False)
    while not stop_event.is_set():
        # バッチ分 latent と補助パラメータを収集
        latent_list = []
        latmask_list = []
        trans_param_list = []
        dconst_list = []
        for _ in range(batch_size):
            z_current = next(latent_gen)  # shape: [1, z_dim]
            latent_list.append(z_current)
            if custom and hasattr(Gs.synthesis, 'input'):
                latmask_i = lmask[frame_idx_local % len(lmask)] if lmask is not None else None
                trans_param_i = trans_params[frame_idx % len(trans_params)] if trans_params is not None else None
                dconst_i = dconst[frame_idx % dconst.shape[0]] if dconst is not None else None
            else:
                latmask_i = None
                trans_param_i = None
                dconst_i = None
            latmask_list.append(latmask_i)
            trans_param_list.append(trans_param_i)
            dconst_list.append(dconst_i)
            frame_idx_local += 1
            frame_idx += 1
        # スタックしてバッチ化（latent_list の各要素は shape [1, z_dim]）
        latent_batch = torch.cat(latent_list, dim=0)  # shape: [batch_size, z_dim]
        # ラベルもバッチ対応（もし label が存在するなら）
        if label is not None:
            label_batch = label.repeat(batch_size, 1)
        else:
            label_batch = None

        with torch.no_grad():
            if custom and hasattr(Gs.synthesis, 'input'):
                # ここでは各補助パラメータをバッチ化できるようにする必要があります
                # ※ もし各パラメータは単一サンプル用なら、リストのままでも内部でループするか、
                #     事前に stack してバッチとして渡すことを検討してください。
                # ここでは簡易的に、各パラメータをリストとして渡す例です。
                out = Gs(latent_batch, label_batch, latmask_list, trans_param_list, dconst_list,
                         truncation_psi=config_args["trunc"], noise_mode='const')
            else:
                out = Gs(latent_batch, label_batch, None, truncation_psi=config_args["trunc"], noise_mode='const')
        # out はバッチ分の画像（shape: [batch_size, H, W, C] と仮定）
        # 各画像をキューに投入
        for i in range(out.shape[0]):
            frame = out[i].permute(1, 2, 0).cpu().numpy()[..., ::-1]  # BGR変換（必要に応じて）
            frame_queue.put(frame, block=True)
