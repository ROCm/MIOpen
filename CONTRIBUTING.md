# MIOpen Contributing Guide
To ensure the quality of the MIOpen code base, the MIOpen team has 
established a code review process to inform developers of the steps 
that are required to shepherd a change-set into the repository.

## Creating a Pull Request
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

## Responsibility of Author
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