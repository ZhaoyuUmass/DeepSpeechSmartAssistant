#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function

import argparse
import numpy as np
import shlex
import subprocess
import sys
import wave
import json
import pyaudio
import struct

import socket, threading

from deepspeech import Model, version
from timeit import default_timer as timer


try:
    from shhlex import quote
except ImportError:
    from pipes import quote


# speech file constant
CHUNK = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000
RECORD_SECONDS = 5
WAVE_OUTPUT_FILENAME = "test.wav"
WIDTH = 2
frames = []

HOST = '10.42.0.1'  # Symbolic name meaning all available interfaces
PORT = 4444  # Arbitrary non-privileged port

BULB_NAME = "bc"

# packet types
types = {"speech": 0,
         "register": 1,
         "ping": 2 # used for test only
         }

packer = struct.Struct('I')
name_packer = struct.Struct('2s')

conn_map = {}
state = 0

class ClientThread(threading.Thread):
    def __init__(self, conn, addr, model):
        threading.Thread.__init__(self)
        self.conn = conn
        self.caddr = addr
        self.model = model
        self.frame = []
        print("New connection added: ", addr)

    def run(self):
        data = self.conn.recv(4)
        req_type = packer.unpack(data)
        if req_type == 0:
            while data:
                data = self.conn.recv(1024)
                self.frame.append(data)

            # b''.join(self.frame)

            wf = wave.open(WAVE_OUTPUT_FILENAME, 'wb')
            wf.setnchannels(CHANNELS)
            wf.setsampwidth(2)
            wf.setframerate(RATE)
            wf.writeframes(b''.join(self.frame))
            wf.close()

            fin = wave.open(WAVE_OUTPUT_FILENAME, 'rb')
            audio = np.frombuffer(fin.readframes(fin.getnframes()), np.int16)
            text = self.model.stt(audio)

            if 'off' in text:
                if conn_map[BULB_NAME] is not None:
                    conn_map[BULB_NAME].send(packer.pack(0))
            elif 'on' in text:
                if conn_map[BULB_NAME] is not None:
                    conn_map[BULB_NAME].send(packer.pack(1))
            else:
                print('No idea what to do')

        elif req_type == 1:
            data = self.conn.recv(2)
            c = name_packer.unpack(data)

            conn_map[c] = self.conn

        elif req_type == 2:
            global state
            if conn_map[BULB_NAME] is not None:
                conn_map[BULB_NAME].send(packer.pack(state))
            state = 1 if(state == 0) else 0

        else:
            print('Unrecognized request type')


def convert_samplerate(audio_path, desired_sample_rate):
    sox_cmd = 'sox {} --type raw --bits 16 --channels 1 --rate {} --encoding signed-integer --endian little --compression 0.0 --no-dither - '.format(quote(audio_path), desired_sample_rate)
    try:
        output = subprocess.check_output(shlex.split(sox_cmd), stderr=subprocess.PIPE)
    except subprocess.CalledProcessError as e:
        raise RuntimeError('SoX returned non-zero status: {}'.format(e.stderr))
    except OSError as e:
        raise OSError(e.errno, 'SoX not found, use {}hz files or install it: {}'.format(desired_sample_rate, e.strerror))

    return desired_sample_rate, np.frombuffer(output, np.int16)


def metadata_to_string(metadata):
    return ''.join(token.text for token in metadata.tokens)


def words_from_candidate_transcript(metadata):
    word = ""
    word_list = []
    word_start_time = 0
    # Loop through each character
    for i, token in enumerate(metadata.tokens):
        # Append character to word if it's not a space
        if token.text != " ":
            if len(word) == 0:
                # Log the start time of the new word
                word_start_time = token.start_time

            word = word + token.text
        # Word boundary is either a space or the last character in the array
        if token.text == " " or i == len(metadata.tokens) - 1:
            word_duration = token.start_time - word_start_time

            if word_duration < 0:
                word_duration = 0

            each_word = dict()
            each_word["word"] = word
            each_word["start_time "] = round(word_start_time, 4)
            each_word["duration"] = round(word_duration, 4)

            word_list.append(each_word)
            # Reset
            word = ""
            word_start_time = 0

    return word_list


def metadata_json_output(metadata):
    json_result = dict()
    json_result["transcripts"] = [{
        "confidence": transcript.confidence,
        "words": words_from_candidate_transcript(transcript),
    } for transcript in metadata.transcripts]
    return json.dumps(json_result, indent=2)



class VersionAction(argparse.Action):
    def __init__(self, *args, **kwargs):
        super(VersionAction, self).__init__(nargs=0, *args, **kwargs)

    def __call__(self, *args, **kwargs):
        print('DeepSpeech ', version())
        exit(0)



def main():
    parser = argparse.ArgumentParser(description='Running DeepSpeech inference.')
    parser.add_argument('--model', required=False, default='deepspeech-0.7.3-models.pbmm',
                        help='Path to the model (protocol buffer binary file)')
    parser.add_argument('--scorer', required=False, default='deepspeech-0.7.3-models.scorer',
                        help='Path to the external scorer file')
    parser.add_argument('--audio', required=False, default='test.wav',
                        help='Path to the audio file to run (WAV format)')
    parser.add_argument('--beam_width', type=int,
                        help='Beam width for the CTC decoder')
    parser.add_argument('--lm_alpha', type=float,
                        help='Language model weight (lm_alpha). If not specified, use default from the scorer package.')
    parser.add_argument('--lm_beta', type=float,
                        help='Word insertion bonus (lm_beta). If not specified, use default from the scorer package.')
    parser.add_argument('--version', action=VersionAction,
                        help='Print version and exits')
    parser.add_argument('--extended', required=False, action='store_true',
                        help='Output string from extended metadata')
    parser.add_argument('--json', required=False, action='store_true',
                        help='Output json from metadata with timestamp of each word')
    parser.add_argument('--candidate_transcripts', type=int, default=3,
                        help='Number of candidate transcripts to include in JSON output')
    parser.add_argument('--port', type=int, default=8080,
                        help='Port number this server binds to')
    args = parser.parse_args()

    print('Loading model from file {}'.format(args.model), file=sys.stderr)
    model_load_start = timer()
    # sphinx-doc: python_ref_model_start
    ds = Model(args.model)
    # sphinx-doc: python_ref_model_stop
    model_load_end = timer() - model_load_start
    print('Loaded model in {:.3}s.'.format(model_load_end), file=sys.stderr)

    if args.beam_width:
        ds.setBeamWidth(args.beam_width)

    desired_sample_rate = ds.sampleRate()

    if args.scorer:
        print('Loading scorer from files {}'.format(args.scorer), file=sys.stderr)
        scorer_load_start = timer()
        ds.enableExternalScorer(args.scorer)
        scorer_load_end = timer() - scorer_load_start
        print('Loaded scorer in {:.3}s.'.format(scorer_load_end), file=sys.stderr)

        if args.lm_alpha and args.lm_beta:
            ds.setScorerAlphaBeta(args.lm_alpha, args.lm_beta)

    fin = wave.open(args.audio, 'rb')
    fs_orig = fin.getframerate()
    if fs_orig != desired_sample_rate:
        print('Warning: original sample rate ({}) is different than {}hz. Resampling might produce erratic speech recognition.'.format(fs_orig, desired_sample_rate), file=sys.stderr)
        fs_new, audio = convert_samplerate(args.audio, desired_sample_rate)
    else:
        audio = np.frombuffer(fin.readframes(fin.getnframes()), np.int16)

    audio_length = fin.getnframes() * (1/fs_orig)
    fin.close()

    print('Running inference.', file=sys.stderr)
    inference_start = timer()
    # sphinx-doc: python_ref_inference_start
    if args.extended:
        print(metadata_to_string(ds.sttWithMetadata(audio, 1).transcripts[0]))
    elif args.json:
        print(metadata_json_output(ds.sttWithMetadata(audio, args.candidate_transcripts)))
    else:
        print(ds.stt(audio))
    # sphinx-doc: python_ref_inference_stop
    inference_end = timer() - inference_start
    print('Inference took %0.3fs for %0.3fs audio file.' % (inference_end, audio_length), file=sys.stderr)


    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind((HOST, PORT))
    while True:
        s.listen(1)
        conn, clientAddress = s.accept()
        newthread = ClientThread(conn, clientAddress, ds)
        newthread.start()


if __name__ == '__main__':
    main()
