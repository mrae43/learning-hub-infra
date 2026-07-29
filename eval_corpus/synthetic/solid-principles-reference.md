# SOLID Principles Reference Guide

## Introduction

SOLID is a mnemonic acronym for five design principles intended to make software designs more understandable, flexible, and maintainable. These principles apply to object-oriented design at the class and module level.

## 1. Single Responsibility Principle (SRP)

A class should have one, and only one, reason to change. This means each class should have a single, well-defined responsibility. When a class has multiple responsibilities, changes to one responsibility may affect the other, making the system fragile.

### Application in Python

In a Python monorepo, SRP means:
- Pydantic models handle validation only
- Repositories handle data access only
- Services handle business logic only
- Route handlers handle HTTP contract only

### Anti-Pattern: God Class

A god class is a class that knows too much or does too much. For example, a UserManager class that validates data, creates database records, sends emails, and logs activity violates SRP. The symptom is a class with many unrelated methods and dependencies.

## 2. Open/Closed Principle (OCP)

Software entities should be open for extension but closed for modification. You should be able to add new functionality without changing existing code. In Python, this is achieved through abstract base classes (ABC), protocols, and dependency injection.

### Implementation Strategy

Define abstract interfaces in a shared package, and implement them in specific packages. Use a factory or registry pattern to select implementations at runtime. Adding a new implementation should not require changing existing code — only registering the new implementation.

### Anti-Pattern: If/Else Chains

Long if/elif/else chains that select behavior based on a type string violate OCP. Adding a new type requires modifying the chain. The fix is to replace the chain with a registry mapping types to implementations.

## 3. Liskov Substitution Principle (LSP)

Subtypes must be substitutable for their base types. If S is a subtype of T, then objects of type T may be replaced with objects of type S without altering the desirable properties of the program.

### Contract Requirements

Subtypes must:
- Accept the same input types (or broader)
- Return the same output types (or narrower)
- Not strengthen preconditions
- Not weaken postconditions
- Not remove exceptions that callers expect

### Testing LSP

Substitution tests verify interchangeability. Create test cases that use the parent type and run them against all implementations. These tests catch contract violations early.

## 4. Interface Segregation Principle (ISP)

Clients should not be forced to depend on interfaces they do not use. Large, fat interfaces should be split into smaller, more specific ones. In Python, protocols are the natural tool for ISP.

### Application

Instead of one large Repository interface with CRUD + search + export methods, define separate Reader, Writer, and Deletable protocols. A read-only service only depends on Reader, not on the full Repository.

## 5. Dependency Inversion Principle (DIP)

High-level modules should not depend on low-level modules. Both should depend on abstractions. Abstractions should not depend on details; details should depend on abstractions.

### Implementation

Dependencies should be injected via constructors or function parameters rather than instantiated directly. In FastAPI, use Depends() for dependency injection. This makes the system testable because dependencies can be swapped with mocks.

### Anti-Pattern: Concrete Instantiation

Instantiating concrete classes inside business logic violates DIP. For example, creating a PostgresUserRepository() inside a service method couples the service to the database implementation. The fix is to accept the repository as a parameter or constructor argument.

## Quick Reference Table

| Principle | Goal | Red Flag | Python Pattern |
|-----------|------|----------|----------------|
| SRP | One reason to change | Class does 3+ jobs | Separate into service/repository/validator |
| OCP | Extend without modifying | Adding if/else chains | Use ABC or Protocol for abstraction |
| LSP | Substitutable types | Subclass breaks parent contract | Ensure type compatibility |
| ISP | Focused contracts | Clients have unused methods | Split fat interfaces |
| DIP | Depend on abstractions | Concrete instantiation in functions | Inject via Depends() or constructor |
