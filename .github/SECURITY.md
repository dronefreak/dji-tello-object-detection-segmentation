# Security Policy

## Supported Versions

| Version | Supported          |
| ------- | ------------------ |
| 2.0.x   | :white_check_mark: |
| < 2.0   | :x:                |

## Reporting a Vulnerability

We take the security of Tello Vision seriously. If you discover a security vulnerability, please follow these steps:

### How to Report

1. **Do NOT create a public GitHub issue** for security vulnerabilities
2. **Email the maintainers directly** or use GitHub's private vulnerability reporting feature
3. **Open a private security advisory** at: `Security` → `Advisories` → `New draft security advisory`

### What to Include

When reporting a vulnerability, please include:

- Description of the vulnerability
- Steps to reproduce the issue
- Potential impact
- Suggested fix (if you have one)
- Your contact information

### Response Timeline

- **Initial Response**: Within 48 hours
- **Status Update**: Within 7 days
- **Fix Timeline**: Depends on severity
  - Critical: 1-7 days
  - High: 7-30 days
  - Medium: 30-90 days
  - Low: Best effort

## Security Considerations

### Drone Operations

⚠️ **This software controls physical drones. Use responsibly.**

- Always maintain visual line of sight
- Test in safe, open environments
- Keep firmware updated
- Follow local aviation regulations
- Never fly near people, airports, or restricted areas

### Network Security

- The DJI Tello creates an **unencrypted WiFi network**
- Video stream is **not encrypted**
- Control commands are sent over **UDP without authentication**
- Assume anyone in range can intercept communications

**Mitigation:**

- Use in controlled environments only
- Don't transmit sensitive information
- Be aware of your surroundings

### Code Execution

This project:

- Downloads and executes ML models from external sources
- Processes video streams (potential for adversarial attacks)
- Uses third-party dependencies with their own security considerations

**Best Practices:**

- Only use models from trusted sources
- Keep dependencies updated: `pip install --upgrade -r requirements.txt`
- Review code before running in production environments
- Use virtual environments to isolate dependencies

### Model Weights

Pre-trained models are downloaded from:

- Ultralytics (YOLOv8): https://github.com/ultralytics/assets/releases
- Detectron2: https://dl.fbaipublicfiles.com/detectron2/

**Verify checksums** if using in security-sensitive applications.

## Known Limitations

### Not Production-Hardened For

- ❌ Mission-critical applications
- ❌ Safety-critical systems
- ❌ Environments requiring formal verification
- ❌ Applications requiring encrypted communications

### Suitable For

- ✅ Research and development
- ✅ Educational purposes
- ✅ Prototyping and experimentation
- ✅ Hobbyist projects

## Dependencies

We rely on several third-party packages. Security issues in dependencies should be reported to:

- **PyTorch**: https://github.com/pytorch/pytorch/security
- **OpenCV**: https://github.com/opencv/opencv/security
- **Ultralytics**: https://github.com/ultralytics/ultralytics/security
- **djitellopy**: https://github.com/damiafuentes/DJITelloPy/security

## Security Updates

Security updates will be:

- Released as patch versions (2.0.x)
- Announced in release notes
- Tagged with `security` label in issues

## Responsible Disclosure

We follow a coordinated disclosure policy:

1. Vulnerability reported privately
2. Issue confirmed and assessed
3. Fix developed and tested
4. Security advisory published
5. Fix released
6. Public disclosure after users have time to update

## Legal

This software is provided "AS IS" without warranty of any kind. Users are responsible for:

- Safe operation of drones
- Compliance with local laws and regulations
- Any damage or injury resulting from use

## Questions?

For non-security questions, use GitHub Discussions or Issues.

For security concerns, contact maintainers privately.

---

**Remember: Safety first. This controls a physical device that can cause injury or property damage if misused.**
