## Contributing

[fork]: https://help.github.com/articles/fork-a-repo/
[pr]: https://help.github.com/articles/creating-a-pull-request/
[style]: https://www.python.org/dev/peps/pep-0008/
[code-of-conduct]: CODE_OF_CONDUCT.md
[azurepipelines]: azure-pipelines.yml
[benchmark]: BENCHMARK.md

Hi there! We're thrilled that you'd like to contribute to this project. Your help is essential for keeping it great.

Contributions to this project are [released](https://help.github.com/articles/github-terms-of-service/#6-contributions-under-repository-license) to the public under the [project's open source license](LICENSE).

Please note that this project is released with a [Contributor Code of Conduct][code-of-conduct]. By participating in this project you agree to abide by its terms.

## Scope

We anticipate that the community will design custom architectures and use frameworks other than Tensorflow.  Furthermore, we anticipate that other datasets beyond the ones provided in this project might be useful.  It is not our intention to integrate the best models and datasets into this repository as a superset of all available ideas.  Rather, we intend to provide baseline approaches and a central place of reference with links to related repositories from the community.  Therefore, we are accepting pull requests for the following items:

- Bug fixes
- Updates to documentation, including links to your project(s) where improvements to the baseline have been made
- Minor improvements to the code

Please open an issue if you are unsure regarding the best course of action.  

## Submitting a pull request

0. [Fork][fork] and clone the repository
0. Configure and install the dependencies: `script/bootstrap`
0. Make sure the tests pass on your machine: see [azure-pipelines.yml][azurepipelines] to see tests we are currently running.
0. Create a new branch: `git checkout -b my-branch-name`
0. Make your change, add tests, and make sure the tests still pass.
0. Push to your fork and [submit a pull request][pr]
0. Pat your self on the back and wait for your pull request to be reviewed and merged.

Here are a few things you can do that will increase the likelihood of your pull request being accepted:

- Follow the [style guide][style].
- Write tests.
- Keep your change as focused as possible. If there are multiple changes you would like to make that are not dependent upon each other, consider submitting them as separate pull requests.
- Write a [good commit message](http://tbaggery.com/2008/04/19/a-note-about-git-commit-messages.html).

## Resources

- [How to Contribute to Open Source](https://opensource.guide/how-to-contribute/)
- [Using Pull Requests](https://help.github.com/articles/about-pull-requests/)
- [GitHub Help](https://help.github.com)
