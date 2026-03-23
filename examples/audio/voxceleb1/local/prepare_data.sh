#!/bin/bash
# Copyright (c) 2023 Shuai Wang (wsstriving@gmail.com)
#               2026 Ke Zhang (kylezhang1118@gmail.com)

stage=-1
stop_stage=-1

single_data_path='./voxceleb/VoxCeleb1/wav/'
mix_data_path='./Libri2Mix/wav16k/min/'

data=data
noise_type=clean
num_spk=2

. tools/parse_options.sh || exit 1

real_data=$(realpath ${data})

if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
  echo "Prepare VoxCeleb1 single-speaker samples.jsonl for online mix"

  dataset=train-vox1
  out_dir="${real_data}/${noise_type}/${dataset}"
  mkdir -p "${out_dir}"

  python local/scan_voxceleb1.py \
    "${single_data_path}" \
    --outfile "${out_dir}/samples.jsonl"

  ln -sf samples.jsonl "${out_dir}/raw.list"
fi

if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
  dataset=train-vox1
  mix_index="${real_data}/${noise_type}/${dataset}/samples.jsonl"
  out_dir="${real_data}/${noise_type}/${dataset}/cues"
  mkdir -p "${out_dir}"

  python local/build_audio_cues_vox1.py \
    --samples_jsonl "${mix_index}" \
    --outfile "${out_dir}/audio.json"

cat > ${data}/${noise_type}/${dataset}/cues.yaml << EOF
cues:
  audio:
    type: raw
    guaranteed: true
    scope: speaker
    policy:
      type: random
      key: spk_id
      resource: ${data}/${noise_type}/${dataset}/cues/audio.json
EOF
fi

if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
  echo "stage 3.0: Prepare the meta files for the datasets (JSONL)"

  for dataset in dev test; do
    echo "Preparing JSONL for" $dataset

    dataset_path=$mix_data_path/$dataset/mix_${noise_type}
    out_dir="${real_data}/${noise_type}/${dataset}"
    mkdir -p "${out_dir}"

    python local/scan_librimix.py \
      "${dataset_path}" \
      --outfile "${out_dir}/samples.jsonl"

    ln -sf samples.jsonl "${out_dir}/raw.list"

  done

  echo "stage 3.1: Build fixed audio cues for eval and test sets from samples.jsonl"

  for dset in dev test; do
  mix_index="${real_data}/${noise_type}/${dset}/samples.jsonl"
  out_dir="${real_data}/${noise_type}/${dset}/cues"
  mkdir -p "${out_dir}"

  # A) Generate speech.json(sanity check)
  python local/build_audio_cues.py \
    --samples_jsonl "${mix_index}" \
    --outfile "${out_dir}/audio.json"

  # B) Download mixture2enrollment
  if [ $num_spk -eq 2 ]; then
    url="https://raw.githubusercontent.com/BUTSpeechFIT/speakerbeam/main/egs/libri2mix/data/wav8k/min/${dset}/map_mixture2enrollment"
  else
    url="https://raw.githubusercontent.com/BUTSpeechFIT/speakerbeam/main/egs/libri3mix/data/wav8k/min/${dset}/map_mixture2enrollment"
  fi

  wget -O "${out_dir}/mixture2enrollment" "$url"

  # C) Generate fixed_enroll.json
  python local/build_fixed_enroll_from_BUT.py \
    --mixture2enrollment "${out_dir}/mixture2enrollment" \
    --speech_json "${out_dir}/audio.json" \
    --outfile "${out_dir}/fixed_enroll.json"

  # D) Generate cues.yaml for dev/test
  cat > ${real_data}/${noise_type}/${dset}/cues.yaml << EOF
cues:
  audio:
    type: raw
    guaranteed: true
    scope: speaker
    policy:
      type: fixed
      key: mix_spk_id
      resource: ${data}/${noise_type}/${dset}/cues/fixed_enroll.json
EOF
  done
fi

if [ ${stage} -le 4 ] && [ ${stop_stage} -ge 4 ]; then
  echo "Download the pre-trained speaker encoders (Resnet34 & Ecapa-TDNN512) from wespeaker..."
  mkdir wespeaker_models
  wget https://wespeaker-1256283475.cos.ap-shanghai.myqcloud.com/models/voxceleb/voxceleb_resnet34.zip
  unzip voxceleb_resnet34.zip -d wespeaker_models
  wget https://wespeaker-1256283475.cos.ap-shanghai.myqcloud.com/models/voxceleb/voxceleb_ECAPA512.zip
  unzip voxceleb_ECAPA512.zip -d wespeaker_models
fi

if [ ${stage} -le 5 ] && [ ${stop_stage} -ge 5 ]; then
  if [ ! -d "${real_data}/raw_data/musan" ]; then
    mkdir -p ${real_data}/raw_data/musan
    #
    echo "Downloading musan.tar.gz ..."
    echo "This may take a long time. Thus we recommand you to download all archives above in your own way first."
    wget --no-check-certificate https://openslr.elda.org/resources/17/musan.tar.gz -P ${real_data}/raw_data
    md5=$(md5sum ${real_data}/raw_data/musan.tar.gz | awk '{print $1}')
    [ $md5 != "0c472d4fc0c5141eca47ad1ffeb2a7df" ] && echo "Wrong md5sum of musan.tar.gz" && exit 1

    echo "Decompress all archives ..."
    tar -xzvf ${real_data}/raw_data/musan.tar.gz -C ${real_data}/raw_data

    rm -rf ${real_data}/raw_data/musan.tar.gz
  fi

  echo "Prepare wav.scp for musan ..."
  mkdir -p ${real_data}/musan
  find -L ${real_data}/raw_data/musan -name "*.wav" | awk -F"/" '{print $(NF-2)"/"$(NF-1)"/"$NF,$0}' >${real_data}/musan/wav.scp

  # Convert all musan data to LMDB
  echo "conver musan data to LMDB ..."
  python tools/make_lmdb.py ${real_data}/musan/wav.scp ${real_data}/musan/lmdb
fi