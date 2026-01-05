"""
BigOcrPdf - Custom Exceptions Module

This module defines custom exception classes for specific error cases
in the BigOcrPdf application.
"""


class BigOcrPdfError(Exception):
    """Base exception for all BigOcrPdf errors.

    All custom exceptions should inherit from this class to allow
    catching any BigOcrPdf-specific error.
    """

    def __init__(self, message: str, details: str | None = None) -> None:
        """Initialize the exception.

        Args:
            message: Human-readable error message
            details: Optional technical details for debugging
        """
        super().__init__(message)
        self.message = message
        self.details = details

    def __str__(self) -> str:
        if self.details:
            return f"{self.message} ({self.details})"
        return self.message


class FileNotFoundError(BigOcrPdfError):
    """Raised when a required file is not found.

    Note: This shadows the builtin FileNotFoundError, use explicit
    builtins.FileNotFoundError if needed.
    """

    def __init__(self, file_path: str, message: str | None = None) -> None:
        """Initialize the exception.

        Args:
            file_path: Path to the file that was not found
            message: Optional custom message
        """
        self.file_path = file_path
        msg = message or f"File not found: {file_path}"
        super().__init__(msg, details=f"path={file_path}")


class InvalidPdfError(BigOcrPdfError):
    """Raised when a PDF file is invalid or corrupted."""

    def __init__(self, file_path: str, reason: str | None = None) -> None:
        """Initialize the exception.

        Args:
            file_path: Path to the invalid PDF file
            reason: Optional reason why the PDF is invalid
        """
        self.file_path = file_path
        self.reason = reason
        msg = f"Invalid PDF file: {file_path}"
        if reason:
            msg += f" - {reason}"
        super().__init__(msg, details=f"path={file_path}")


class OcrProcessingError(BigOcrPdfError):
    """Raised when OCR processing fails."""

    def __init__(
        self,
        file_path: str,
        reason: str | None = None,
        exit_code: int | None = None,
    ) -> None:
        """Initialize the exception.

        Args:
            file_path: Path to the file that failed processing
            reason: Optional reason for the failure
            exit_code: Optional exit code from the OCR process
        """
        self.file_path = file_path
        self.reason = reason
        self.exit_code = exit_code

        msg = f"OCR processing failed for: {file_path}"
        if reason:
            msg += f" - {reason}"

        details = f"path={file_path}"
        if exit_code is not None:
            details += f", exit_code={exit_code}"

        super().__init__(msg, details=details)


class ConfigurationError(BigOcrPdfError):
    """Raised when there's a configuration-related error."""

    def __init__(self, setting_name: str | None = None, reason: str | None = None) -> None:
        """Initialize the exception.

        Args:
            setting_name: Optional name of the problematic setting
            reason: Optional reason for the error
        """
        self.setting_name = setting_name
        self.reason = reason

        if setting_name:
            msg = f"Configuration error for '{setting_name}'"
        else:
            msg = "Configuration error"

        if reason:
            msg += f": {reason}"

        super().__init__(msg)


class LanguageNotAvailableError(BigOcrPdfError):
    """Raised when a requested OCR language is not available."""

    def __init__(self, language: str, available_languages: list[str] | None = None) -> None:
        """Initialize the exception.

        Args:
            language: The language that was requested
            available_languages: Optional list of available languages
        """
        self.language = language
        self.available_languages = available_languages or []

        msg = f"OCR language '{language}' is not available"
        if available_languages:
            msg += f". Available languages: {', '.join(available_languages[:5])}"
            if len(available_languages) > 5:
                msg += f" (and {len(available_languages) - 5} more)"

        super().__init__(msg)


class OutputPathError(BigOcrPdfError):
    """Raised when there's an issue with the output path."""

    def __init__(self, output_path: str, reason: str | None = None) -> None:
        """Initialize the exception.

        Args:
            output_path: The problematic output path
            reason: Optional reason for the error
        """
        self.output_path = output_path
        self.reason = reason

        msg = f"Output path error: {output_path}"
        if reason:
            msg += f" - {reason}"

        super().__init__(msg, details=f"path={output_path}")


class PermissionDeniedError(BigOcrPdfError):
    """Raised when there's a permission issue accessing a file or directory."""

    def __init__(self, path: str, operation: str = "access") -> None:
        """Initialize the exception.

        Args:
            path: Path that couldn't be accessed
            operation: The operation that was attempted (read, write, etc.)
        """
        self.path = path
        self.operation = operation

        msg = f"Permission denied: cannot {operation} '{path}'"
        super().__init__(msg, details=f"path={path}, operation={operation}")


class ProcessTimeoutError(BigOcrPdfError):
    """Raised when a process times out."""

    def __init__(self, file_path: str, timeout_seconds: float) -> None:
        """Initialize the exception.

        Args:
            file_path: Path to the file being processed
            timeout_seconds: The timeout duration that was exceeded
        """
        self.file_path = file_path
        self.timeout_seconds = timeout_seconds

        msg = f"Processing timed out after {timeout_seconds}s for: {file_path}"
        super().__init__(msg, details=f"timeout={timeout_seconds}s")


class QueueError(BigOcrPdfError):
    """Raised when there's an error with the processing queue."""

    def __init__(self, reason: str, queue_size: int | None = None) -> None:
        """Initialize the exception.

        Args:
            reason: Reason for the queue error
            queue_size: Optional current queue size
        """
        self.reason = reason
        self.queue_size = queue_size

        msg = f"Queue error: {reason}"
        details = None
        if queue_size is not None:
            details = f"queue_size={queue_size}"

        super().__init__(msg, details=details)


class ValidationError(BigOcrPdfError):
    """Raised when input validation fails."""

    def __init__(
        self,
        field: str,
        value: str | None = None,
        reason: str | None = None,
    ) -> None:
        """Initialize the exception.

        Args:
            field: Name of the field that failed validation
            value: Optional value that failed validation
            reason: Optional reason for the validation failure
        """
        self.field = field
        self.value = value
        self.reason = reason

        msg = f"Validation error for '{field}'"
        if reason:
            msg += f": {reason}"

        super().__init__(msg)


class DependencyError(BigOcrPdfError):
    """Raised when a required dependency is missing or incompatible."""

    def __init__(
        self,
        dependency: str,
        required_version: str | None = None,
        installed_version: str | None = None,
    ) -> None:
        """Initialize the exception.

        Args:
            dependency: Name of the missing or incompatible dependency
            required_version: Optional required version
            installed_version: Optional installed version
        """
        self.dependency = dependency
        self.required_version = required_version
        self.installed_version = installed_version

        if installed_version and required_version:
            msg = (
                f"Dependency '{dependency}' version mismatch: "
                f"required {required_version}, found {installed_version}"
            )
        elif required_version:
            msg = f"Missing dependency: {dependency} >= {required_version}"
        else:
            msg = f"Missing dependency: {dependency}"

        super().__init__(msg)


# Exception hierarchy summary:
# BigOcrPdfError (base)
# ├── FileNotFoundError
# ├── InvalidPdfError
# ├── OcrProcessingError
# ├── ConfigurationError
# ├── LanguageNotAvailableError
# ├── OutputPathError
# ├── PermissionDeniedError
# ├── ProcessTimeoutError
# ├── QueueError
# ├── ValidationError
# └── DependencyError
