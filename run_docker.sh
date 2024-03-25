docker run -it \
    --gpus=all \
    --rm \
    --volume "/home/nikisim/Mag_diplom/CORL:/workspace/" \
    --name corl \
    corl bash
