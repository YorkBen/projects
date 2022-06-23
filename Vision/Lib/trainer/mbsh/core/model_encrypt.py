

__author__ = 'Steven Zheng'

import os
import io
import struct

from tensorflow.python.keras.models import model_from_json
# 存在No module named 'tensorflow.python.keras._impl'错误，恢复原来的包导入
# from tensorflow.python.keras._impl.keras.engine.topology import load_weights_from_hdf5_group
# from tensorflow.python.keras.engine.saving import load_weights_from_hdf5_group
# from tensorflow.python.keras.engine.saving.hdf5_format import load_weights_from_hdf5_group
# from tensorflow.python.keras.saving.hdf5_format import load_weights_from_hdf5_group
# modified by pengxiang 20220510

from mbsh import logger

import h5py
# from Crypto.Cipher import AES
from Cryptodome.Cipher import AES

# key len must be 16, 24, or 32 bytes
# g_encry_key = 'endoangel--wuhan'.encode('utf-8')
g_encry_key = 'endoangel--J&^5#'.encode('utf-8')
g_encry_tmp_enc = '.e'


def save_enc_model(json_file_path, weight_file_path):
    encrypt_file(json_file_path)
    encrypt_file(weight_file_path)
    logger.info("save model success ,path=%s,%s" % (json_file_path, weight_file_path))


def load_enc_model(json_file_path, weight_file_path):
    model = None

    try:
        in_filename = json_file_path + g_encry_tmp_enc
        json_str = decrypt_file(in_filename, file_handle=False)
        model = model_from_json(json_str)

        in_filename = weight_file_path + g_encry_tmp_enc
        dec_file = decrypt_file(in_filename)
        load_weights(model, dec_file)

        logger.info('load weights %s' % in_filename)
    except Exception as e:
        logger.error("load model fail %s,%s" % (in_filename, e))

    return model


def load_weights(model, f, reshape=False):
    f = f['model_weights']
    # load_weights_from_hdf5_group(f, model.layers, reshape=reshape)
    # load_weights_from_hdf5_group(f, model.layers)
    # modified by pengxiang 20220510
    # ************************This place should be focused*************
    model.load_weights(f)


def encrypt_file(in_filename, key=g_encry_key, chunksize=64 * 1024):
    """
    加密文件
    """
    from Crypto.Util.Padding import pad, unpad
    out_filename_temp = in_filename + g_encry_tmp_enc
    iv = os.urandom(16)
    encryptor = AES.new(key, AES.MODE_CBC, iv)
    filesize = os.path.getsize(in_filename)
    with open(in_filename, 'rb') as infile:
        with open(out_filename_temp, 'wb') as outfile:
            outfile.write(struct.pack('<Q', filesize))
            outfile.write(iv)
            pos = 0
            while pos < filesize:
                chunk = infile.read(chunksize)
                pos += len(chunk)
                if pos == filesize:
                    chunk = pad(chunk, AES.block_size)
                outfile.write(encryptor.encrypt(chunk))

    logger.info("export encrypt file :%s" % out_filename_temp)


def decrypt_file(in_filename, file_handle=True, key=g_encry_key, chunksize=64 * 1024):
    from Crypto.Util.Padding import pad, unpad
    with open(in_filename, 'rb') as infile:
        filesize = struct.unpack('<Q', infile.read(8))[0]
        iv = infile.read(16)
        encryptor = AES.new(key, AES.MODE_CBC, iv)

        bio = io.BytesIO()
        encrypted_filesize = os.path.getsize(in_filename)
        pos = 8 + 16  # the filesize and IV.

        while pos < encrypted_filesize:
            chunk = infile.read(chunksize)
            pos += len(chunk)
            chunk = encryptor.decrypt(chunk)
            if pos == encrypted_filesize:
                chunk = unpad(chunk, AES.block_size)

            bio.write(chunk)

    return h5py.File(bio, mode='r') if file_handle else bio # str(bio.getvalue(), 'utf-8')
