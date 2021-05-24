# MIOpen Contributing Guide
To ensure the quality of the MIOpen code base, the MIOpen team has 
established a code review process to inform developers of the steps 
that are required to shepherd a change-set into the repository.

#### Table Of Contents

[How to get started](#How-to-get-started)

[How do I contribute?](#how-do-i-contribute)
  * [Reporting Issues](#reporting-issues)
  * [Creating a Pull Request](#Creating-a-Pull-Request)

[Responsibility of the Author](#Responsibility-of-the-Author)

[Responsibility of the Reviewer](#Responsibility-of-the-Reviewer)

[Passing CI](#Passing-CI)

[The Review](#the-review)
## How to get started
MIOpen is AMD’s deep learning primitives library which
provides highly optimized, and hand-tuned implementations of
different operators such as convolution, batch normalization,
pooling, softmax, activation and layers for Recurrent Neural
Networks (RNNs), used in both training and inference.

The easiest way to get started is to check out [MIOpen documentation](https://rocmsoftwareplatform.github.io/MIOpen/doc/html/) and [MIOpen: An Open Source Library For Deep Learning Primitives](https://arxiv.org/pdf/1910.00078.pdf).

All contributions you make will be under the [MIT Software License](LICENSE.txt). 
## How do I contribute
### Reporting Issues
We use [GitHub Issues](https://github.com/ROCmSoftwarePlatform/MIOpen/issues) to track public **bugs** and **enhancement requests**.

If you have found an issue, please check [MIOpen documentation](https://rocmsoftwareplatform.github.io/MIOpen/doc/html/) to see if it is a bug or new feature enhancement which is not supported in the latest version of MIOpen:

#### Bugs
Please follow the template below to report any bugs found in MIOpen:

1. Description: ***Please be clear and descriptive***
2. How to Reproduce the issue:
* Hardware Information:
* Docker environment or Software Version:
* Expected behavior:
* Actual behavior:
3. Any additional information:

#### Enhancement Requests
Please follow the template below to report any enhancement requests for MIOpen:

1. Description: ***Please be clear and descriptive***
2. Value and Motivation:
* Feature and functionalities enabled:
* Any alternatives:
3. Any additional information:

The author must set labels (and assigns a miliestone) according to his/her own understanding.

Other contributors can change these values if they disagree. That being said, 
adding a small comment explaining the motivation is highly recommended. 
In this way, we keep the process flexible while cultivating mutual understanding.

[**Note**] Most likely, the labels like "bug", "feature" or "complexity*" 
would not be changed. However, "value*" or "urgency*" might be from mutual 
understanding.
### Creating a Pull Request
No changes are allowed to be directly committed to the develop 
branch of the MIOpen repository. All authors are required to 
develop their change sets on a separate branch and then create 
a pull request (PR) to merge their changes into the develop branch.

Once a PR has been created, a developer must choose two reviewers 
to review the changes made. The first reviewer should be a 
technical expert in the portion of the library that the changes 
are being made in. You can find a list of these experts in 
[MIOpen Issue #789](https://github.com/ROCmSoftwarePlatform/MIOpen/issues/789)
. The second reviewer should be a peer reviewer. This reviewer 
can be any other MIOpen developer.

## Responsibility of the Author
The author of a PR is responsible for:
 * Writing clear, well documented code
 * Meeting expectations of code quality
 * Verifying that the changes do not break current functionality
 * Writing tests to ensure code coverage
 * Report on the impact to performance

## Responsibility of the Reviewer
Each reviewer is responsible for verifying that the changes are 
clearly written in keeping with the coding styles of the library, 
are documented in a way that future developers will be able to 
understand the intent of the added functionality, and will 
maintain or improve the overall quality of the codebase.

Reviewer's task checklist:
1. Has the PR passed [necessary CI](https://github.com/ROCmSoftwarePlatform/MIOpen/pull/932#discussion_r634835432)?
2. Does the PR consist of a well-organized sequence of small commits, 
each of which is designed to make one specific feature or fix 
(and ideally should be able to pass CI testing)?
3. Does the PR only include a reviewable amount of changes? Or it is a 
consolidation of already reviewed small batches? e.g. break it into smaller 
testable and reviewable tasks instead of a huge chunk at once.
4. Does the PR have sufficient documentation and easy to read and understand, 
feasible for test and future maintainence, related docs already in place? 
e.g. revise or add to 
[MIOpen documentation](https://rocmsoftwareplatform.github.io/MIOpen/doc/html/) 
if API or functionality has changed?
5. For bugfixes and new features, new regression test created and included in CI,
 or some other holistic test pipeline?
6. Is every PR associated with a ticket or issue number for tracking purposes?

## Passing CI
The most critical component of the PR process is the CI testing. 
All PRs must pass the CI in order to be considered for merger. 
Reviewers may choose to defer their review until the CI testing 
has passed. 

## The Review
During the review, reviewers will look over the changes and make 
suggestions or requests for changes.

In order to assist the reviewer in prioritizing their efforts, 
authors can take the following actions:

* Set the urgency and value labels
* Set the milestone where the changes need to be delivered
* Describe the testing procedure and post the measured effect of 
  the change
* Remind reviewers via email if a PR needs attention
* If a PR needs to be reviewed as soon as possible, explain to 
  the reviewers why a review may need to take priority
