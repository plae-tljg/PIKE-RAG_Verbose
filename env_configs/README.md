# Environment Configuration

This folder contains dummy configuration files to help users bypass setup errors.

## Setup

1. Copy the `.env.dummy` file to the root directory as `.env`:
   ```bash
   cp env_configs/.env.dummy ../.env
   ```

2. Edit `.env` and add your actual API keys and credentials.

3. The `.env` file will be gitignored to protect your credentials.

## Notes

- Keep dummy files in this directory
- Never commit your actual `.env` file
