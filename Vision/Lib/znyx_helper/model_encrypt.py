__author__ = 'Steven Zheng'

import os
import io
import struct
import json
from tensorflow.python.keras.models import model_from_json, load_model, save_model, model_from_config

try:
    from tensorflow.python.keras.saving.hdf5_format import load_weights_from_hdf5_group
except ImportError:
    from tensorflow.python.keras._impl.keras.engine.topology import load_weights_from_hdf5_group


import h5py
from Cryptodome.Cipher import AES
from Cryptodome.Util.Padding import pad, unpad

# key len must be 16, 24, or 32 bytes
g_encry_key = 'endoangel--J&^5#'.encode('utf-8')
g_encry_tmp_enc = '.e'


def save_enc_model(json_file_path, weight_file_path):
    encrypt_file(json_file_path)
    encrypt_file(weight_file_path)
    print("save model success ,path=%s,%s" % (json_file_path, weight_file_path))


def load_enc_model(json_file_path, weight_file_path):
    model = None

    try:
        in_filename = json_file_path + g_encry_tmp_enc
        json_str = decrypt_file(in_filename, file_handle=False)
        model = model_from_json(json_str)

        in_filename = weight_file_path + g_encry_tmp_enc
        dec_file = decrypt_file(in_filename)
        load_weights(model, dec_file)

        print('load weights %s' % in_filename)
    except Exception as e:
        print("load model fail %s,%s" % (in_filename, e))

    return model


def load_weights(model, f, reshape=False):
    f = f['model_weights']
    # load_weights_from_hdf5_group(f, model.layers, reshape=reshape)
    load_weights_from_hdf5_group(f, model.layers)


def encrypt_file(in_filename, key=g_encry_key, chunksize=64 * 1024):
    """
    加密文件
    """
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

    print("export encrypt file :%s" % out_filename_temp)
    return out_filename_temp


def decrypt_file(in_filename, file_handle=True, key=g_encry_key, chunksize=64 * 1024):
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

    return h5py.File(bio) if file_handle else str(bio.getvalue(), 'utf-8')


def load_src_model(model_file, weight_file=None):
    """
    @:param weight_file 是否独立存储json和weights
    """
    model = None
    try:
        if weight_file:
            _file = open(model_file, 'r')
            model = model_from_json(_file.read())
            model.load_weights(weight_file)
        else:
            load_model(model_file, False)

        print('load weights %s' % weight_file)
    except:
        print("load model fail %s" % weight_file)
    return model


def save_encrypt_model(model, out_file_path):
    save_model(model, out_file_path)

    en_file = encrypt_file(out_file_path)
    return en_file


def load_encrypt_model(out_file_path):
    in_filename = out_file_path
    print('load entry file %s ' % in_filename)
    f = decrypt_file(in_filename,file_handle=True)
    model_config = f.attrs.get('model_config')
    if model_config is None:
        raise ValueError('No model found in config file.')
    model_config = json.loads(model_config.decode('utf-8'))
    model = model_from_config(model_config)
    load_weights_from_hdf5_group(f['model_weights'], model.layers)

    return model
