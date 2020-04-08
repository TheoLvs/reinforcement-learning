#!/usr/bin/python
# -*- coding: utf-8 -*-

from __future__ import print_function

import json
import time
import os
from collections import namedtuple


# Python 2/3 compatibility
# Python 2:
try:
    from urlparse import urlparse
    from urllib import urlencode
    from urllib2 import urlopen, Request, HTTPError
except:
    pass

# Python 3:
try:
    from time import process_time
    from urllib.parse import urlparse, urlencode
    from urllib.request import urlopen, Request
    from urllib.error import HTTPError
except:
    pass

import sys
# Python 2:
if sys.version_info < (3, 0):
    def process_time():
        return time.clock()
    def input(str):
        return raw_input(str)

# Python 3, backward compatibility with unicode test
if sys.version_info >= (3, 0):
    unicode = type(str)

version = '1.0.0'
submitt_url = \
    'https://www.coursera.org/api/onDemandProgrammingScriptSubmissions.v1'

Metadata = namedtuple("Metadata", ['assignment_key', 'name', 'part_data'])
Part = namedtuple("Part", ['id', 'input_file', 'solver_file', 'name'])


def load_metadata(metadata_file_name='_coursera'):
    '''
    Parses an assignment metadata file

    Args:
        metadata_file_name (str): location of the metadata file

    Returns:
        metadata as a named tuple structure
    '''

    if not os.path.exists(metadata_file_name):
        print('metadata file "%s" not found' % metadata_file_name)
        quit()

    try:
        with open(metadata_file_name, 'r') as metadata_file:
            url = metadata_file.readline().strip()
            name = metadata_file.readline().strip()
            part_data = []
            for line in metadata_file.readlines():
                if ',' in line:
                    line_parts = line.split(',')
                    line_parts = [x.strip() for x in line_parts]
                    assert(len(line_parts) == 4)
                    part_data.append(Part(*line_parts))
            if len(url) <= 0:
                print('Empty url in _coursera file: %s' % metadata_file_name)
                quit()
            if len(name) <= 0:
                print('Empty assignment name in _coursera file: %s' % metadata_file_name)
                quit()
    except Exception as e:
        print('problem parsing assignment metadata file')
        print('exception message:')
        print(e)
        quit()

    return Metadata(url, name, part_data)


def part_prompt(problems):
    '''
    Prompts the user for which parts of the assignment they would like to 
    submit.

    Args:
        problems:  a list of assignment problems

    Returns:
        the selected subset of problems
    '''

    count = 1
    print('Hello! These are the assignment parts that you can submit:')
    for i, problem in enumerate(problems):
        print(str(count) + ') ' + problem.name)
        count += 1
    print('0) All')

    part_text = input('Please enter which part(s) you want to submit (0-%d): ' % (count-1))
    selected_problems = []
    selected_models = []

    for item in part_text.split(','):
        try:
            i = int(item)
        except:
            print('Skipping input "' + item + '".  It is not an integer.')
            continue

        if i >= count or i < 0:
            print('Skipping input "' + item + '".  It is out of the valid range (0-%d).' % (count-1))
            continue

        if i == 0:
            selected_problems.extend(problems)
            continue

        if i <= len(problems):
            selected_problems.append(problems[i-1])

    if len(selected_problems) <= 0:
        print('No valid assignment parts identified.  Please try again. \n')
        return part_prompt(problems)
    else:
        return selected_problems


def compute(metadata, solver_file_override=None):
    '''
    Determines which assignment parts the student would like to submit.
    Then computes his/her answers to those assignment parts

    Args:
        metadata:  the assignment metadata
        solver_file_override:  an optional model file to override the metadata 
            default

    Returns:
        a dictionary of results in the format Coursera expects
    '''

    if solver_file_override is not None:
        print('Overriding solver file with: '+solver_file_override)

    selected_problems = part_prompt(metadata.part_data)

    results = {}

    #submission needs empty dict for every assignment part
    results.update({prob_data.id : {} for prob_data in metadata.part_data})

    for problem in selected_problems:
        if solver_file_override != None:
            solver_file = solver_file_override
        else:
            solver_file = problem.solver_file
        
        if not os.path.isfile(solver_file):
            print('Unable to locate assignment file "%s" in the current working directory.' % solver_file)
            continue

        # if a relative path is given, add that patth to system path so import will work
        if os.path.sep in solver_file:
            split = solver_file.rfind(os.path.sep)
            path = solver_file[0:split]
            file_name = solver_file[split+1:]
            sys.path.insert(0, path)
            solver_file = file_name

        submission = output(problem.input_file, solver_file)
        if submission != None:
            results[problem.id] = {'output':submission}

    print('\n== Computations Complete ...')

    return results


def load_input_data(file_location):
    with open(file_location, 'r') as input_data_file:
        input_data = ''.join(input_data_file.readlines())
    return input_data


def output(input_file, solver_file):
    '''
    Attempts to execute solve_it locally on a given input file.

    Args:
        input_file: the assignment problem data of interest
        solver_file: a python file containing the solve_it function

    Returns:
        the submission string in a format that the grader expects
    '''

    try:
        pkg = __import__(solver_file[:-3]) # remove '.py' extension
        if not hasattr(pkg, 'solve_it'):
            print('the solve_it() function was not found in %s' % solver_file)
            quit()
    except ImportError:
        print('import error with python file "%s".' % solver_file)
        quit()


    solution = ''

    start = process_time()
    try:
        solution = pkg.solve_it(load_input_data(input_file))
    except Exception as e:
        print('the solve_it(input_data) method from solver.py raised an exception')
        print('try testing it with python ./solver.py before running this submission script')
        print('exception message:')
        print(str(e))
        print('')
        return 'Local Exception =('
    end = process_time()

    if not (isinstance(solution, str) or isinstance(solution, unicode)):
        print('Warning: the solver did not return a string.  The given object will be converted with the str() method.')
        solution = str(solution)

    print('Submitting: ')
    print(solution)

    return solution.strip() + '\n' + str(end - start)


def login_dialog(assignment_key, results, credentials_file_location = '_credentials'):
    '''
    Requests Coursera login credentials from the student and submits the 
    student's solutions for grading

    Args:
        assignment_key: Coursera's assignment key
        results: a dictionary of results in Cousera's format
        credentials_file_location: a file location where login credentials can 
            be found
    '''

    success = False
    tries = 0

    while not success:

        # stops infinate loop when credentials file is incorrect 
        if tries <= 0:
            login, token = login_prompt(credentials_file_location)
        else:
            login, token = login_prompt('')

        code, response = submit_solution(assignment_key, login, token, results)

        print('\n== Coursera Response ...')
        #print(code)
        print(response)
        
        if code != 401:
            success = True
        else:
            print('\ntry logging in again')
        tries += 1

def login_prompt(credentials_file_location):
    '''
    Attempts to load credentials from a file, if that fails asks the user.
    Returns:
        the user's login and token
    '''
    
    if os.path.isfile(credentials_file_location):
        try:
            with open(credentials_file_location, 'r') as metadata_file:
                login = metadata_file.readline().strip()
                token = metadata_file.readline().strip()
                metadata_file.close()
        except:
            login, token = basic_prompt()
    else:
        login, token = basic_prompt()
    return login, token


def basic_prompt():
    '''
    Prompt the user for login credentials. 
    Returns:
        the user's login and token
    '''
    login = input('User Name (e-mail address): ')
    token = input('Submission Token (from the assignment page): ')
    return login, token


def submit_solution(assignment_key, email_address, token, results):
    '''
    Sends the student's submission to Coursera for grading via the submission 
    API.

    Args:
        assignment_key: Coursera's assignment key
        email_address: the student's email
        token: the student's assignment token
        results: a dictionary of results in Cousera's format

    Returns:
        the https response code and a feedback message
    '''

    print('\n== Connecting to Coursera ...')
    print('Submitting %d of %d parts' % 
        (sum(['output' in v for k,v in results.items()]), len(results)))

    # build json datastructure
    parts = {}
    submission = {
        'assignmentKey': assignment_key,  
        'submitterEmail': email_address,  
        'secret': token,
        'parts': results
    }

    # send submission
    req = Request(submitt_url)
    req.add_header('Cache-Control', 'no-cache')
    req.add_header('Content-type', 'application/json')

    try:
        res = urlopen(req, json.dumps(submission).encode('utf8'))
    except HTTPError as e:
        response = json.loads(e.read().decode('utf8'))

        if 'details' in response and response['details'] != None and \
            'learnerMessage' in response['details']:
            return e.code, response['details']['learnerMessage']
        else:
            return e.code, 'Unexpected response code, please contact the ' \
                               'course staff.\nDetails: ' + response['message']

    code = res.code
    response = json.loads(res.read().decode('utf8'))

    if code >= 200 and code <= 299:
        return code, 'Your submission has been accepted and will be ' \
                     'graded shortly.'

    return code, 'Unexpected response code, please contact the '\
                 'course staff.\nDetails: ' + response


def main(args):
    '''
    1) Reads a metadata file to customize the submission process to 
    a particular assignment.  
    2) The compute the student's answers to the assignment parts.
    3) Submits the student's answers for grading.

    Provides the an option for saving the submissions locally.  This is very 
    helpful when testing the assignment graders.

    Args:
        args: CLI arguments from an argparse parser
    '''

    # needed so that output can import from the cwd
    sys.path.append(os.getcwd())

    if args.metadata is None:
        metadata = load_metadata()
    else:
        print('Overriding metadata file with: '+args.metadata)
        metadata = load_metadata(args.metadata)

    print('==\n== '+metadata.name+' Solution Submission \n==')
    
    # compute dialog
    results = compute(metadata, args.override)

    if sum(['output' in v for k,v in results.items()]) <= 0:
        return

    # store submissions if requested
    if args.record_submission == True:
        print('Recording submission as files')
        for sid, submission in results.items():
            if 'output' in submission:
                directory = '_'+sid
                if not os.path.exists(directory):
                    os.makedirs(directory)

                submission_file_name = directory+'/submission.sub'
                print('  writting submission file: '+submission_file_name)
                with open(submission_file_name,'w') as submission_file:
                    submission_file.write(submission['output'])
                    submission_file.close()
        return

    # submit dialog
    if args.credentials is None:
        login_dialog(metadata.assignment_key, results)
    else:
        print('Overriding credentials file with: '+args.credentials)
        login_dialog(metadata.assignment_key, results, args.credentials)



import argparse
def build_parser():
    '''
    Builds an argument parser for the CLI

    Returns:
        parser: an argparse parser
    '''

    parser = argparse.ArgumentParser(
        description='''The submission script for Discrete Optimization 
            assignments on the Coursera Platform.''',
        epilog='''Please file bugs on github at: 
        https://github.com/discreteoptimization/assignment/issues. If you 
        would like to contribute to this tool's development, check it out at: 
        https://github.com/discreteoptimization/assignment'''
    )

    parser.add_argument('-v', '--version', action='version', 
        version='%(prog)s '+version)

    parser.add_argument('-o', '--override', 
        help='overrides the python source file specified in the \'_coursera\' file')

    parser.add_argument('-m', '--metadata', 
        help='overrides the \'_coursera\' metadata file')

    parser.add_argument('-c', '--credentials', 
        help='overrides the \'_credentials\' credentials file')

    parser.add_argument('-rs', '--record_submission', 
        help='records the submission(s) as files', action='store_true')

    return parser


if __name__ == '__main__':
    parser = build_parser()
    main(parser.parse_args())

