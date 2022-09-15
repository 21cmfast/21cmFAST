============
Contributing
============

Contributions are welcome, and they are greatly appreciated! Every
little bit helps, and credit will always be given.

Bug reports/Feature Requests/Feedback/Questions
===============================================
It is incredibly helpful to us when users report bugs, unexpected behaviour, or request
features. You can do the following:

* `Report a bug <https://github.com/21cmFAST/21cmFAST/issues/new?template=bug_report.md>`_
* `Request a Feature <https://github.com/21cmFAST/21cmFAST/issues/new?template=feature_request.md>`_
* `Ask a Question <https://github.com/21cmFAST/21cmFAST/issues/new?template=question.md>`_

When doing any of these, please try to be as succinct, but detailed, as possible, and use
a "Minimum Working Example" whenever applicable.

Documentation improvements
==========================

``21cmFAST`` could always use more documentation, whether as part of the
official ``21cmFAST`` docs, in docstrings, or even on the web in blog posts,
articles, and such. If you do the latter, take the time to let us know about it!

High-Level Steps for Development
================================

This is an abbreviated guide to getting started with development of ``21cmFAST``,
focusing on the discrete high-level steps to take. See our
`notes for developers <https://21cmfast.readthedocs.org/en/latest/notes_for_developers>`_
for more details about how to get around the ``21cmFAST`` codebase and other
technical details.

There are two avenues for you to develop ``21cmFAST``. If you plan on making significant
changes, and working with ``21cmFAST`` for a long period of time, please consider
becoming a member of the 21cmFAST GitHub organisation (by emailing any of the owners
or admins). You may develop as a member or as a non-member.

The difference between members and non-members only applies to the first step
of the development process.

Note that it is highly recommended to work in an isolated python environment with
all requirements installed from ``environment_dev.txt``. This will also ensure that
pre-commit hooks will run that enforce the ``black`` coding style. If you do not
install these requirements, you must manually run ``black`` before committing your changes,
otherwise your changes will likely fail continuous integration.

As a *member*:

1. Clone the repo::

    git clone git@github.com:21cmFAST/21cmFAST.git

As a *non-member*:

1. First fork ``21cmFAST <https://github.com/21cmFAST/21cmFAST>``_
   (look for the "Fork" button), then clone the fork locally::

    git clone git@github.com:your_name_here/21cmFAST.git

The following steps are the same for both *members* and *non-members*:

2. Install a fresh new isolated environment::

       conda create -n 21cmfast python=3
       conda activate 21cmfast

3. Install the *development* requirements for the project::

    conda env update -f environment_dev.yml

4. Install 21cmFAST. See `Installation <./installation.html>`_ for more details.::

    pip install -e .

4. Install pre-commit hooks::

    pre-commit install

5. Create a branch for local development::

    git checkout -b name-of-your-bugfix-or-feature

   Now you can make your changes locally. **Note: as a member, you _must_ do step 5. If you
   make changes on master, you will _not_ be able to push them**.

6. When you're done making changes, run ``pytest`` to check that your changes didn't
   break things. You can run a single test or subset of tests as well (see pytest docs)::

    pytest

7. Commit your changes and push your branch to GitHub::

    git add .
    git commit -m "Your detailed description of your changes."
    git push origin name-of-your-bugfix-or-feature

   Note that if the commit step fails due to a pre-commit hook, *most likely* the act
   of running the hook itself has already fixed the error. Try doing the ``add`` and
   ``commit`` again (up, up, enter). If it's still complaining, manually fix the errors
   and do the same again.

8. Submit a pull request through the GitHub website.

Pull Request Guidelines
-----------------------

If you need some code review or feedback while you're developing the code just make the
pull request. You can mark the PR as a draft until you are happy for it to be merged.
