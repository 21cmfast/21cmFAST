============
Contributing
============

Contributions are welcome, and they are greatly appreciated! Every
little bit helps, and credit will always be given.

Bug reports/Feature Requests/Feedback/Questions
===============================================
It is incredibly helpful to us when users report bugs, unexpected behaviour, or request
features. You can do the following:

    * `Report a bug <https://github.com/21cmFAST/21cmFAST/issues/new?template=bug_report.md>`
    * `Request a Feature <https://github.com/21cmFAST/21cmFAST/issues/new?template=feature_request.md>`
    * `Ask a Question <https://github.com/21cmFAST/21cmFAST/issues/new?template=question.md>`

When doing any of these, please try to be as succinct, but detailed, as possible, and use
a "Minimum Working Example" whenever applicable.

Documentation improvements
==========================

21cmFAST could always use more documentation, whether as part of the
official 21cmFAST docs, in docstrings, or even on the web in blog posts,
articles, and such.

Development
===========

This is an abbreviated guide to contributing, focusing on the discrete steps to take.
See https://21cmfast.readthedocs.org/en/latest/notes_for_developers for more details.

There are two avenues for you to develop `21cmFAST`. If you plan on making significant
changes, and working with `21cmFAST` for a long period of time, please consider
becoming a member of the 21cmFAST GitHub organisation (by emailing any of the owners
or admins). You may develop as a member or as a non-member.

The difference between members and non-members only applies to the first two steps
of the development process.

Note that it is highly recommended to work in an isolated python environment with
all requirements installed from ``requirements_dev.txt``. This will also ensure that
pre-commit hooks will run that enforce the ``black`` coding style. If you do not
install these requirements, you must manually run black before committing your changes,
otherwise your changes will likely fail continuous integration.

As a member:

1. Clone the repo::

    git clone git@github.com:21cmFAST/21cmFAST.git

As a non-member:

1. First fork `21cmFAST <https://github.com/21cmFAST/21cmFAST>`_
   (look for the "Fork" button), then clone the fork locally::

    git clone git@github.com:your_name_here/21cmFAST.git

The following steps are the same for both members and non-members:

2. Create a branch for local development::

    git checkout -b name-of-your-bugfix-or-feature

   Now you can make your changes locally. **Note: you _must_ do this step. If you
   make changes on master, you will _not_ be able to push them, as a member**.

3. When you're done making changes, run all the checks, doc builder and spell checker
   with `tox <http://tox.readthedocs.io/en/latest/install.html>`_ one command::

    tox

4. Commit your changes and push your branch to GitHub::

    git add .
    git commit -m "Your detailed description of your changes."
    git push origin name-of-your-bugfix-or-feature

5. Submit a pull request through the GitHub website.

Pull Request Guidelines
-----------------------

If you need some code review or feedback while you're developing the code just make the
pull request. You can mark the PR as a draft until you are happy for it to be merged.

Tips
----

To run a subset of tests::

    tox -e envname -- py.test -k test_myfeature

To run all the test environments in *parallel* (you need to ``pip install detox``)::

    detox
